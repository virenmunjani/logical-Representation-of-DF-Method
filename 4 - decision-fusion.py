def apply_fusion(behavioral_predictions, anomaly_predictions, weights):
    fused_decision = []
    for bd_pred, ad_pred, weight in zip(behavioral_predictions, anomaly_predictions, weights):
        fused_decision.append(bd_pred * weight + ad_pred * (1 - weight))  # Weighted fusion
    return fused_decision
