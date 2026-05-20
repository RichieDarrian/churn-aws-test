"""
Streamlit UI for the Churn classifier hosted on SageMaker.
"""

import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "churn-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: dict) -> dict:
    runtime = get_runtime_client()

    payload = {"instances": [features]}

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )

    return json.loads(response["Body"].read().decode("utf-8"))


st.title("Customer Churn Prediction")
st.write("Enter customer information below to predict churn via SageMaker.")

age = st.number_input("Age", min_value=0, max_value=100, value=30)
gender = st.radio("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (years)", min_value=0, max_value=100, value=5)
usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=100, value=10)
support_calls = st.number_input("Support Calls", min_value=0, max_value=10, value=1)
payment_delay = st.number_input("Payment Delay (months)", min_value=0, max_value=30, value=0)
subscription_type = st.radio("Subscription Type", ["Standard", "Premium", "Basic"])
contract_length = st.radio("Contract Length", ["Annual", "Quarterly", "Monthly"])
total_spend = st.number_input("Total Spend", min_value=0, value=1000)
last_interaction = st.number_input("Last Interaction (months)", min_value=0, max_value=30, value=1)

if st.button("Predict", type="primary"):

    features = {
        "Age": int(age),
        "Gender": gender,
        "Tenure": int(tenure),
        "Usage Frequency": int(usage_frequency),
        "Support Calls": int(support_calls),
        "Payment Delay": int(payment_delay),
        "Subscription Type": subscription_type,
        "Contract Length": contract_length,
        "Total Spend": int(total_spend),
        "Last Interaction": int(last_interaction),
    }

    try:
        result = invoke_endpoint(features)

    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )

    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")

    else:
        probs = result["probabilities"][0]
        labels = ["No Churn", "Churn"]
    
        predicted_label = labels[result["predictions"][0]]
    
        st.success(f"Predicted Churn: **{predicted_label}**")
        st.write("Class probabilities:")
    
        import pandas as pd
    
        chart_df = pd.DataFrame(
            {"Probability": probs},
            index=labels
        )
    
        st.bar_chart(chart_df)
