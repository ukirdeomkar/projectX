import streamlit as st
import pandas as pd
import pickle
import base64

# Function to apply the model and return DataFrame with predictions
def apply_model(model, df):
    predictions = model.predict(df)
    df['Prediction'] = predictions
    return df

def main():
    st.title('Toolwear Detection App')

    # Sidebar for file upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read uploaded CSV file into DataFrame
        df = pd.read_csv(uploaded_file)


        # Load pre-trained model
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Drop specified columns
        columns_to_drop = ['target', 'Machining_Process']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Apply model
        new_df = apply_model(model, df.copy())


        # Calculate percentage of 'yes' (1) values in 'Prediction' column
        percent_yes = (new_df['Prediction'] == 1).mean() * 100

        # Choose text color based on percentage
        color = "red" if percent_yes >= 50 else "green"

        # Format percentage text
        percent_text = f"<span style='font-size:32px;color:{color}'><b>{percent_yes:.2f}%</b></span>"
        
        # Display the label and percentage text at the top of the page
        st.markdown("## Chances of worn : " + percent_text , unsafe_allow_html=True)
        # st.markdown(percent_text, unsafe_allow_html=True)

        # Original Table display
        st.write("Original DataFrame:")
        st.write(df)

        # Ouput Table display 
        st.write("DataFrame with Predictions:")
        st.write(new_df)


        # Button to export CSV
        if st.button('Export Predictions to CSV'):
            new_df.to_csv('predictions.csv', index=False)
            st.success("CSV file exported successfully!")

            # Offer download link
            csv = new_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Convert DataFrame to bytes
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
