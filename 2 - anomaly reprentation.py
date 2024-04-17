anomaly_predictions = anomaly_classifier.predict(test_data)

def decision_fusion_behavioral_anomaly(behavioral_predictions, anomaly_predictions):
    fused_decision = []
    for bd_pred, ad_pred in zip(behavioral_predictions, anomaly_predictions):
        if bd_pred == 1 or ad_pred == 1:
            fused_decision.append(1)  # Ransomware
        else:
            fused_decision.append(0)  # Benign
    return fused_decision