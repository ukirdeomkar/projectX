import streamlit as st
import pandas as pd
import pickle
import base64
import io
import openpyxl

# Function to apply the model and return DataFrame with predictions
def apply_model(model, df):
    predictions = model.predict(df)
    df['Prediction'] = predictions
    return df

# Function to highlight rows based on prediction values - using darker red color
def highlight_worn(row):
    return ['background-color: #ff6666' if row['Prediction'] == 1 else '' for _ in row]

def main():
    st.title('Toolwear Detection App')

    # Sidebar for file upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read uploaded CSV file into DataFrame
        df = pd.read_csv(uploaded_file)


        # Load pre-trained model
        with open('xgb_model_new.pkl', 'rb') as f:
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

        # Ouput Table display with colored rows for prediction = 1
        st.write("DataFrame with Predictions:")
        styled_df = new_df.style.apply(highlight_worn, axis=1)
        st.write(styled_df)
        
        # Add explanatory note about red highlighted rows with updated message and dark mode compatible styling
        st.markdown("""
        <div style="border: 1px solid rgba(250, 100, 100, 0.5); padding: 15px; border-radius: 5px; margin-top: 10px; background-color: rgba(255, 100, 100, 0.1);">
            <span style="color: #ff6666; font-weight: bold;">■</span> <b>Note:</b> Rows highlighted in red indicates potential tool wear in the given specific line. These parameters need to be changed in order to avoid tool wear.
        </div>
        """, unsafe_allow_html=True)


        # Export option
        st.write("## Export Option")
        
        # Excel Export Button (with styling)
        if st.button('Export to Excel'):
            # Create a new DataFrame with an extra column to mark rows to highlight
            excel_df = new_df.copy()
            
            # Create a BytesIO object to hold the Excel file
            output = io.BytesIO()
            
            # Create an Excel writer
            writer = pd.ExcelWriter(output, engine='openpyxl')
            
            # Write DataFrame to Excel
            excel_df.to_excel(writer, index=False, sheet_name='Predictions')
            
            # Access the worksheet
            worksheet = writer.sheets['Predictions']
            
            # Apply darker red fill to rows with Prediction = 1
            for idx, row in excel_df.iterrows():
                if row['Prediction'] == 1:
                    # Excel rows are 1-indexed and have a header row, so add 2
                    row_idx = idx + 2
                    for col_idx in range(1, len(excel_df.columns) + 1):
                        cell = worksheet.cell(row=row_idx, column=col_idx)
                        cell.fill = openpyxl.styles.PatternFill(start_color="FF6666", end_color="FF6666", fill_type="solid")
            
            # Save the workbook
            writer.close()
            
            # Get the value of the BytesIO buffer
            b64_excel = base64.b64encode(output.getvalue()).decode()
            
            # Generate download link
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="predictions_with_highlighting.xlsx">Download Excel file with highlighting</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("Excel file created!")

if __name__ == '__main__':
    main()
