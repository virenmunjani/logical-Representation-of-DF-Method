def assign_weights(behavioral_confidence, anomaly_confidence):
    weights = []
    for bd_conf, ad_conf in zip(behavioral_confidence, anomaly_confidence):
        # Example: Higher confidence, higher weight
        weight = bd_conf + ad_conf
        weights.append(weight)
    return weights