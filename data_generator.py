import numpy as np
import pandas as pd

np.random.seed(42)

def generate_data(n_samples=600):
    data = pd.DataFrame()

    # Bias & Fairness
    data['sensitive_attributes'] = np.random.randint(0, 2, n_samples)
    data['demographic_imbalance'] = np.random.randint(0, 3, n_samples)
    data['historical_bias'] = np.random.randint(0, 2, n_samples)
    data['fairness_testing'] = np.random.randint(0, 2, n_samples)
    data['proxy_risk'] = np.random.randint(0, 3, n_samples)

    # Privacy
    data['pii_usage'] = np.random.randint(0, 3, n_samples)
    data['anonymization'] = np.random.randint(0, 2, n_samples)
    data['retention_policy'] = np.random.randint(0, 2, n_samples)
    data['third_party_data'] = np.random.randint(0, 2, n_samples)
    data['encryption'] = np.random.randint(0, 2, n_samples)

    # Transparency
    data['model_type'] = np.random.randint(0, 3, n_samples)
    data['explanation_available'] = np.random.randint(0, 2, n_samples)
    data['documentation_quality'] = np.random.randint(0, 3, n_samples)
    data['audit_conducted'] = np.random.randint(0, 2, n_samples)

    # Deployment
    data['automation_level'] = np.random.randint(0, 3, n_samples)
    data['human_review'] = np.random.randint(0, 2, n_samples)
    data['impact_severity'] = np.random.randint(0, 3, n_samples)
    data['appeals_mechanism'] = np.random.randint(0, 2, n_samples)

    # Risk score calculation
    risk_score = (
        data['sensitive_attributes'] * 12 +
        data['demographic_imbalance'] * 8 +
        data['historical_bias'] * 10 +
        (1 - data['fairness_testing']) * 10 +
        data['proxy_risk'] * 6 +
        data['pii_usage'] * 8 +
        (1 - data['anonymization']) * 8 +
        (1 - data['retention_policy']) * 5 +
        data['third_party_data'] * 5 +
        (1 - data['encryption']) * 6 +
        data['model_type'] * 8 +
        (1 - data['explanation_available']) * 7 +
        data['documentation_quality'] * 5 +
        (1 - data['audit_conducted']) * 6 +
        data['automation_level'] * 10 +
        (1 - data['human_review']) * 12 +
        data['impact_severity'] * 15 +
        (1 - data['appeals_mechanism']) * 8
    )

    # Add controlled noise
    risk_score = risk_score + np.random.randint(-2, 3, n_samples)

    # Normalize to 0–100
    risk_score = np.clip(risk_score / risk_score.max() * 100, 0, 100)

    # Map to categories
    conditions = [
    risk_score <= 25,
    (risk_score > 25) & (risk_score <= 55),
    (risk_score > 55) & (risk_score <= 75),
    risk_score > 75
]

    categories = ['Low', 'Moderate', 'High', 'Critical']

    data['risk_category'] = np.select(conditions, categories,default='Low')

    return data


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/synthetic_ai_governance_data.csv", index=False)
    print("Dataset generated successfully.")