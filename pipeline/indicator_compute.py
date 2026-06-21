"""
Stage 3: Indicator Computation
Computes the five DCTII sub-indicators from Delta-T and ancillary data:
  1. Delta-T_day
  2. Delta-T_night
  3. Heat Island Area (km2)
  4. WBGT-weighted Population Exposed
  5. Waste Heat Flux (W/m2)

Critical design decisions (from critical analysis):
  - PUE -> Waste Heat Flux chain is fully explicit with documented equations.
  - Tier-3 sites (unknown PUE) use a data-driven PUE imputation model
    rather than a single default, with uncertainty propagated into waste heat flux.
  - Default PUE values are documented and justified:
    Modern (post-2018): 1.2, Older (pre-2018): 1.5, Unknown tier-3: imputed.

Maps to GOII's: curated signal -> per-hub features
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger("dctii.indicator_compute")

# ---------------------------------------------------------------------------
# PUE defaults and waste heat physics
# ---------------------------------------------------------------------------

# PUE values by facility vintage (documented justification):
# - Modern (post-2018): industry average ~1.2 per Uptime Institute 2023 survey
# - Older (pre-2018): industry average ~1.5 per EPA Energy Star DC benchmarks
# - PUE range in practice: 1.03 (hyperscale best) to 2.5+ (legacy enterprise)
PUE_DEFAULT_MODERN = 1.2    # post-2018 facilities
PUE_DEFAULT_OLDER = 1.5     # pre-2018 facilities
PUE_UNCERTAINTY_RANGE = {
    1: 0.05,   # Tier 1 (verified): +/- 0.05
    2: 0.15,   # Tier 2 (estimated): +/- 0.15
    3: 0.40,   # Tier 3 (unknown, imputed): +/- 0.40 (can exceed 100% uncertainty in Q)
}


def get_default_pue(activation_year: Optional[int]) -> float:
    """Return default PUE based on facility vintage."""
    if activation_year is None:
        return PUE_DEFAULT_OLDER
    return PUE_DEFAULT_MODERN if activation_year >= 2018 else PUE_DEFAULT_OLDER


class CoolingType(str, Enum):
    """Cooling system classification for sensible/latent heat split."""
    AIR_COOLED = "air_cooled"           # ~95% sensible, ~5% latent
    TOWER_COOLED = "tower_cooled"       # ~60% sensible, ~40% latent (evaporative)
    UNKNOWN = "unknown"                 # Default to air-cooled (conservative)


# Sensible heat fraction by cooling type
SENSIBLE_FRACTION = {
    CoolingType.AIR_COOLED: 0.95,
    CoolingType.TOWER_COOLED: 0.60,
    CoolingType.UNKNOWN: 0.95,  # conservative default
}

# Load factor: actual IT load as fraction of nameplate capacity.
# capacity_mw in site_registry is nameplate, not actual utilization.
# Typical range: 0.4-0.8; industry average ~0.6.
DEFAULT_LOAD_FACTOR = 0.6
LOAD_FACTOR_UNCERTAINTY = 0.15  # +/- 0.15 for unknown utilization

# Capacity ramp-up: data centers don't reach full utilization immediately.
# Industry data (Uptime Institute, JLL) shows 5-7 year ramp to steady state.
# Logistic S-curve: ramp(t) = 1 / (1 + exp(-k*(t - t_mid)))
# k=1.0, t_mid=3 → yr1≈12%, yr2≈27%, yr3≈50%, yr4≈73%, yr5≈88%, yr7≈98%
RAMP_K = 1.0          # steepness of logistic curve
RAMP_MIDPOINT = 3.0   # years to 50% utilization
RAMP_FLOOR = 0.15     # minimum utilization in first year (commissioning load)

# PUE evolution: facilities improve efficiency over time through equipment
# upgrades, hot/cold aisle containment, and operational optimization.
# Uptime Institute surveys: industry average PUE fell from ~1.65 (2010) to
# ~1.55 (2024). Individual facilities improve ~0.8-1.5% per year.
# Floor: best-in-class hyperscale PUE ≈ 1.05 (theoretical minimum ~1.0).
PUE_IMPROVEMENT_RATE = 0.010  # ~1.0% annual improvement
PUE_FLOOR = 1.05              # minimum achievable PUE

# Regional population density estimates (persons/km²) from census data.
# Sources: US Census 2020, StatCan 2021 — suburban/peri-urban densities
# around typical data center zones (industrial/suburban corridors).
REGIONAL_POP_DENSITY = {
    "NOVA": 950.0,    # Northern Virginia — dense suburban (Loudoun/Prince William)
    "PHX":  320.0,    # Phoenix metro — suburban sprawl (Mesa/Chandler)
    "HOU":  580.0,    # Houston — suburban industrial (Katy/Sugar Land corridors)
    "CTX":  210.0,    # Central Texas — exurban (San Marcos/New Braunfels)
    "TOR":  850.0,    # Toronto GTA — suburban (Markham/Vaughan)
    "MTL":  720.0,    # Montreal — suburban (Beauharnois/Vaudreuil)
}
POP_GROWTH_RATE = 0.012  # ~1.2% annual growth (US/Canada suburban average)


def compute_ramp_factor(year: int, activation_year: Optional[int]) -> float:
    """Compute capacity utilization ramp-up factor based on facility age.

    Data centers take 5-7 years to reach steady-state utilization.
    Uses a logistic S-curve clamped to [RAMP_FLOOR, 1.0].
    For facilities activated before our data window (pre-2015 with data
    starting 2015), assumes full ramp-up already achieved.
    """
    if activation_year is None:
        return 1.0  # unknown activation → assume mature
    years_since = year - int(activation_year)
    if years_since <= 0:
        return RAMP_FLOOR  # shouldn't happen (activation guard), but safe
    if years_since >= 8:
        return 1.0  # fully ramped
    raw = 1.0 / (1.0 + np.exp(-RAMP_K * (years_since - RAMP_MIDPOINT)))
    return max(RAMP_FLOOR, min(1.0, raw))


def evolve_pue(base_pue: float, year: int, activation_year: Optional[int]) -> float:
    """Compute year-specific PUE accounting for efficiency improvements over time.

    Data center operators continuously optimize PUE through equipment upgrades,
    airflow management, and cooling improvements. Newer facilities start more
    efficient; older ones improve more aggressively from higher baselines.

    Model: PUE(t) = max(PUE_FLOOR, base_PUE × (1 - rate)^years_since_activation)
    """
    if activation_year is None or base_pue <= PUE_FLOOR:
        return base_pue
    years_since = max(0, year - int(activation_year))
    improved = base_pue * ((1.0 - PUE_IMPROVEMENT_RATE) ** years_since)
    return max(PUE_FLOOR, improved)


def estimate_population_density(region_code: str, year: int, base_year: int = 2020) -> float:
    """Estimate population density for a region and year.

    Uses regional baseline densities from census data and applies
    a compound annual growth rate for temporal extrapolation.
    """
    base_density = REGIONAL_POP_DENSITY.get(region_code, 400.0)
    years_from_base = year - base_year
    return base_density * ((1.0 + POP_GROWTH_RATE) ** years_from_base)


def compute_waste_heat_flux(
    capacity_mw: float,
    pue: float,
    footprint_km2: float,
    confidence_tier: int = 2,
    load_factor: float = DEFAULT_LOAD_FACTOR,
    cooling_type: CoolingType = CoolingType.UNKNOWN,
) -> Tuple[float, float, float, float]:
    """
    Compute waste heat flux from electrical capacity, PUE, and footprint.

    Physics:
      Actual IT load:     P_IT = capacity_mw * load_factor
      Total power input:  P_total = P_IT * PUE
      Waste heat:         Q_waste = P_IT * (PUE - 1)
      Flux density:       q = Q_waste / A_footprint  [W/m2]

    Where:
      capacity_mw = nameplate electrical capacity (NOT actual IT load)
      load_factor = actual utilization fraction (default 0.6, range 0.4-0.8)
      PUE = Power Usage Effectiveness (total facility / IT load)
      A_footprint = facility ground area in m2

    Sensible/latent split (parameterized by cooling_type):
      - Air-cooled: ~95% sensible, ~5% latent  (most common)
      - Tower-cooled: ~60% sensible, ~40% latent (Houston, Toronto common)
      - Unknown: defaults to air-cooled (conservative for thermal impact)

    Uncertainty sources (combined):
      - PUE uncertainty: tier-dependent (+/- 0.05 to +/- 0.40)
      - Load factor uncertainty: +/- 0.15 for unknown utilization
      - Combined via quadrature: sqrt(dQ/dPUE^2 * sigma_PUE^2 + dQ/dLF^2 * sigma_LF^2)

    Returns:
      (q_total_wm2, q_sensible_wm2, q_latent_wm2, q_uncertainty_wm2)
    """
    if footprint_km2 <= 0 or capacity_mw <= 0 or pue < 1.0:
        return (0.0, 0.0, 0.0, 0.0)

    footprint_m2 = footprint_km2 * 1e6
    p_it_w = capacity_mw * 1e6 * load_factor
    q_waste_w = p_it_w * (pue - 1.0)
    q_flux_wm2 = q_waste_w / footprint_m2

    # Sensible/latent split by cooling type
    f_sensible = SENSIBLE_FRACTION.get(cooling_type, 0.95)
    q_sensible = q_flux_wm2 * f_sensible
    q_latent = q_flux_wm2 * (1.0 - f_sensible)

    # Uncertainty propagation (quadrature of PUE and load factor contributions)
    pue_sigma = PUE_UNCERTAINTY_RANGE.get(confidence_tier, 0.40)
    lf_sigma = LOAD_FACTOR_UNCERTAINTY
    # dQ/dPUE = P_IT / A = capacity * LF / A
    dq_dpue = capacity_mw * 1e6 * load_factor / footprint_m2
    # dQ/dLF = capacity * (PUE-1) / A
    dq_dlf = capacity_mw * 1e6 * (pue - 1.0) / footprint_m2
    q_uncertainty = np.sqrt((dq_dpue * pue_sigma) ** 2 + (dq_dlf * lf_sigma) ** 2)

    return (q_flux_wm2, q_sensible, q_latent, q_uncertainty)


def compute_indicators(site_id: str, year: int) -> dict:
    """Compute all five sub-indicators for a site-year."""
    raise NotImplementedError("To be implemented in Stage 3")


def compute_all_indicators(
    delta_t_monthly: pd.DataFrame,
    heat_area_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all 5 sub-indicators for each site × year.
    
    Indicators:
      1. delta_t_day: Annual mean daytime Delta-T (°C)
      2. delta_t_night: Annual mean nighttime Delta-T (°C)
      3. affected_ring_area_km2: Heat island area proxy
      4. population_exposed_base: Affected area × population density
      5. waste_heat_flux_wm2: Engineering-based heat rejection estimate
    
    Supports wide format input (delta_t_day_c and delta_t_night_c in same row).
    Returns DataFrame with one row per site × year.
    """
    if delta_t_monthly.empty:
        logger.warning("No Delta-T data for indicator computation")
        return pd.DataFrame()

    results = []

    # Use preferred sensor (MODIS_MOD11A1 for consistency) or all if not available
    preferred_sensors = ["MODIS_MOD11A1", "MODIS_MYD11A1"]
    if "sensor_id" in delta_t_monthly.columns:
        preferred = delta_t_monthly[delta_t_monthly["sensor_id"].isin(preferred_sensors)]
        if not preferred.empty:
            delta_t_monthly = preferred

    # ── Method correction factor ────────────────────────────────────────
    # ring_difference captures background UHI + facility effect, while
    # cem_weighted isolates the facility-specific signal.  Literature and
    # cross-method comparison in our data show ring_difference ΔT is
    # ~2–3× CEM ΔT.  We apply a 0.4 discount to ring_difference so that
    # stored indicators approximate the facility-attributable component.
    RING_DIFF_DISCOUNT = 0.4

    for (site_id, year), group in delta_t_monthly.groupby(["site_id", "year"]):
        site_info = sites_df[sites_df["site_id"] == site_id]
        if site_info.empty:
            continue
        site = site_info.iloc[0]

        # Skip years before facility activation (no valid data pre-construction)
        activation_year = site.get("activation_year")
        if activation_year and year <= int(activation_year):
            continue

        # Determine method for this site's data
        is_ring = False
        if "method" in group.columns:
            site_method = group["method"].iloc[0]
            is_ring = site_method == "ring_difference"

        # 1. Delta-T day (wide format: delta_t_day_c column directly)
        day_valid = group[group["delta_t_day_c"].notna()]
        delta_t_day = float(day_valid["delta_t_day_c"].mean()) if not day_valid.empty else None
        if delta_t_day is not None and is_ring:
            delta_t_day *= RING_DIFF_DISCOUNT

        # 2. Delta-T night (wide format: delta_t_night_c column directly)
        night_valid = group[group["delta_t_night_c"].notna()]
        delta_t_night = float(night_valid["delta_t_night_c"].mean()) if not night_valid.empty else None
        if delta_t_night is not None and is_ring:
            delta_t_night *= RING_DIFF_DISCOUNT

        # 3. Heat island area
        if heat_area_df.empty:
            affected_area = 0.0
        else:
            area_row = heat_area_df[
                (heat_area_df["site_id"] == site_id) &
                (heat_area_df["year"] == year)
            ]
            affected_area = float(area_row["affected_ring_area_km2"].mean()) if not area_row.empty else 0.0

        # 4. Population exposed — use regional density estimate if covariates empty
        region_code = site.get("region_code", "")
        pop_data = covariates_df[
            (covariates_df["site_id"] == site_id) &
            (covariates_df["zone_name"] == "footprint")
        ] if not covariates_df.empty and "zone_name" in covariates_df.columns else pd.DataFrame()
        if not pop_data.empty and pop_data["population_density"].iloc[0]:
            pop_density = float(pop_data["population_density"].iloc[0])
        else:
            pop_density = estimate_population_density(region_code, year)
        population_exposed = affected_area * pop_density

        # 5. Waste heat flux — apply capacity ramp-up and PUE evolution by year
        act_yr = site.get("activation_year")
        base_pue = site.get("pue_estimate") or get_default_pue(act_yr)
        pue = evolve_pue(base_pue, year, act_yr)
        capacity = site.get("capacity_mw") or 0
        footprint = site.get("footprint_km2") or 0.01
        lf = site.get("load_factor") or DEFAULT_LOAD_FACTOR
        ramp = compute_ramp_factor(year, act_yr)
        effective_lf = lf * ramp
        ct_str = site.get("cooling_type") or "unknown"
        try:
            ct = CoolingType(ct_str)
        except ValueError:
            ct = CoolingType.UNKNOWN

        q_total, q_sens, q_lat, q_unc = compute_waste_heat_flux(
            capacity_mw=capacity, pue=pue, footprint_km2=footprint,
            load_factor=effective_lf, cooling_type=ct,
        )

        # Monthly observation counts
        n_day = int(day_valid["n_clear_days"].sum()) if not day_valid.empty and "n_clear_days" in day_valid.columns else 0
        n_night = int(night_valid["n_clear_nights"].sum()) if not night_valid.empty and "n_clear_nights" in night_valid.columns else 0
        n_months_reliable = 0
        if "reliability_flag" in group.columns:
            n_months_reliable = int(
                group[group["reliability_flag"] == "RELIABLE"].shape[0]
            )

        results.append({
            "site_id": site_id,
            "region_code": site.get("region_code", ""),
            "year": year,
            "delta_t_day_c": delta_t_day,
            "delta_t_night_c": delta_t_night,
            "affected_ring_area_km2": affected_area,
            "population_exposed_base": population_exposed,
            "waste_heat_flux_wm2": q_total,
            "waste_heat_sensible_wm2": q_sens,
            "waste_heat_latent_wm2": q_lat,
            "waste_heat_uncertainty_wm2": q_unc,
            "n_day_obs": n_day,
            "n_night_obs": n_night,
            "n_months_reliable": n_months_reliable,
            "estimation_method": group["method"].iloc[0] if "method" in group.columns else "unknown",
        })

    df = pd.DataFrame(results)
    logger.info(f"Computed indicators for {len(df)} site-years")
    return df


def compute_heat_island_area(
    site_id: str,
    year: int,
    threshold_c: float = 0.5,
    pixel_area_m2: float = 10000.0,
) -> float:
    """
    Count pixels where Delta-T > threshold within R_effective.
    Returns area in km2.

    Computed for multiple thresholds: {0.2, 0.5, 1.0} C.
    Primary threshold is 0.5 C (used in DCTII composite).
    """
    raise NotImplementedError("To be implemented in Stage 3")


def compute_population_exposed(
    site_id: str,
    year: int,
    wbgt_weight: bool = True,
) -> float:
    """
    Overlay heat island area with WorldPop 100m grid.
    Sum population within pixels where Delta-T > threshold.

    WBGT weighting:
      f(WBGT) = 1 + (WBGT - 25) / 10
      Applied as multiplicative weight when wbgt_weight=True.
      WBGT computed from ERA5 (temperature, dewpoint, wind) via Liljegren model.

    For WBGT < 25 C (most Canadian winter months), f(WBGT) < 1,
    reducing the population exposure weight appropriately.
    """
    raise NotImplementedError("To be implemented in Stage 3")


def impute_pue_from_features(
    delta_t_night: float,
    footprint_km2: float,
    climate_zone: str,
    activation_year: Optional[int],
) -> Tuple[float, float]:
    """
    ML-based PUE imputation for tier-3 sites with unknown PUE.

    Regression model trained on tier-1 sites with known PUE values.
    Features: nighttime Delta-T magnitude, facility footprint, climate zone,
    activation year.

    Training data sources: utility filings, operator disclosures, EPA data,
    Uptime Institute surveys.

    Returns: (imputed_pue, imputation_uncertainty)
    """
    raise NotImplementedError("To be implemented in Stage 3 — ML PUE imputation")
