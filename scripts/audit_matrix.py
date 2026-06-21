"""Quick audit of training matrix by estimation method."""
import sys
sys.path.insert(0, ".")
from pipeline.predict_train import build_training_matrix, engineer_features

df = build_training_matrix()
print(df.groupby('estimation_method')[['label_delta_t_day','label_delta_t_night']].agg(['mean','std','count']))
