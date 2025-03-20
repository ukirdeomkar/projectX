import streamlit as st
import pandas as pd
import pickle
import base64
import io
import openpyxl

# Set page config for wider layout
st.set_page_config(
    page_title="Tool Wear Detection System",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #ff4b4b;
        margin-bottom: 2rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #ff6b6b;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .welcome-text {
        font-size: 1.2rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    div[data-testid="stFileUploader"] {
        width: 100%;
    }
    .stButton > button {
        background: linear-gradient(145deg, #ff4b4b, #ff6b6b) !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(145deg, #ff6b6b, #ff4b4b) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.25) !important;
        color: white !important;
    }
    .export-button {
        background: linear-gradient(145deg, #ff4b4b, #ff6b6b);
        color: white !important;
        padding: 12px 24px;
        border-radius: 8px;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        max-width: 300px;
        margin: 0.5rem 0;
        border: none;
        cursor: pointer;
    }
    .export-button:hover {
        background: linear-gradient(145deg, #ff6b6b, #ff4b4b);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.25);
        color: white !important;
        text-decoration: none;
    }
    .export-button svg {
        stroke: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to apply the model and return DataFrame with predictions
def apply_model(model, df):
    predictions = model.predict(df)
    df['Prediction'] = predictions
    return df

# Function to highlight rows based on prediction values - using darker red color
def highlight_worn(row):
    return ['background-color: #ff6666' if row['Prediction'] == 1 else '' for _ in row]

def process_data(uploaded_file):
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
    
    return df, new_df

def main():
    # Main title with custom styling
    st.markdown('<h1 class="main-header">🔧 Tool Wear Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize session state for upload visibility
    if 'show_upload' not in st.session_state:
        st.session_state.show_upload = True
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'new_df' not in st.session_state:
        st.session_state.new_df = None

    # Create three columns with the middle one being wider for the upload section
    left_col, center_col, right_col = st.columns([1, 2, 1])

    with center_col:
        if st.session_state.show_upload:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/gears.svg" 
                     width="100" 
                     style="filter: invert(50%) sepia(50%) saturate(1000%) hue-rotate(320deg);">
            </div>
            """, unsafe_allow_html=True)
            
            if 'uploaded_file' not in st.session_state:
                st.markdown('<p class="welcome-text">Welcome to the Tool Wear Detection System. Upload your machining parameters data to begin analysis.</p>', unsafe_allow_html=True)
            
            # File uploader with custom styling
            st.markdown('<h3 style="color: #ff4b4b; text-align: center; margin-bottom: 1rem;">Upload Data File</h3>', unsafe_allow_html=True)
            
            # Add the file uploader
            uploaded_file = st.file_uploader(
                "Drag and drop your CSV file here or click to browse",
                type=["csv"],
                help="Upload a CSV file containing machining parameters"
            )
            
            # Add visual container around the uploader
            st.markdown("""
            <style>
            [data-testid="stFileUploader"] {
                border: 2px rgba(255, 75, 75, 0.3);
                border-radius: 15px;
                padding: 20px;
                background-color: rgba(255, 75, 75, 0.1);
            }
            [data-testid="stFileUploader"] > section {
                min-height: 100px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if uploaded_file is not None:
                # Process the data and store in session state
                st.session_state.df, st.session_state.new_df = process_data(uploaded_file)
                st.session_state.uploaded_file = uploaded_file
                st.session_state.show_upload = False
                st.rerun()

    # Show toggle upload section button only if data has been uploaded
    if 'uploaded_file' in st.session_state and not st.session_state.show_upload:
        col1, col2, col3 = st.columns([4, 1, 4])
        with col2:
            if st.button('Upload New File'):
                st.session_state.show_upload = True
                st.rerun()

    # Show analysis if file is uploaded and upload section is hidden
    if 'uploaded_file' in st.session_state and not st.session_state.show_upload:
        # Use DataFrames from session state
        df = st.session_state.df
        new_df = st.session_state.new_df

        # Calculate percentage of 'yes' (1) values in 'Prediction' column
        percent_yes = (new_df['Prediction'] == 1).mean() * 100

        # Display analysis results in a card-like container
        st.markdown(f"""
        <div style="background-color: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
            <h2 style="color: #333333; margin-bottom: 20px;">Analysis Results</h2>
            <div style="background-color: {'#ffebeb' if percent_yes >= 50 else '#e6ffe6'}; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: {'#ff4b4b' if percent_yes >= 50 else '#28a745'}; margin-bottom: 10px;">Tool Wear Probability</h3>
                <p style="font-size: 2.5rem; font-weight: bold; color: {'#ff4b4b' if percent_yes >= 50 else '#28a745'}; margin: 0;">{percent_yes:.2f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Data Display Section
        st.markdown('<h2 class="sub-header">📊 Detailed Analysis</h2>', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📈 Predictions", "🔍 Original Data"])
        
        with tab1:
            st.markdown("### Prediction Results")
            styled_df = new_df.style.apply(highlight_worn, axis=1)
            st.write(styled_df)
            
            # Add explanatory note about red highlighted rows
            st.markdown("""
            <div style="border: 1px solid rgba(250, 100, 100, 0.5); padding: 15px; border-radius: 5px; margin-top: 10px; background-color: rgba(255, 100, 100, 0.1);">
                <span style="color: #ff6666; font-weight: bold;">■</span> <b>Note:</b> Rows highlighted in red indicates potential tool wear in the given specific line. These parameters need to be changed in order to avoid tool wear.
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Original Dataset")
            st.write(df)

        # Export Section with Container
        st.markdown("""
        <div style="
            border: 1px solid rgba(255, 75, 75, 0.2);
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem 0;
            background: linear-gradient(145deg, rgba(17, 17, 17, 0.3) 0%, rgba(17, 17, 17, 0.2) 100%);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h2 style="
                color: #ff4b4b;
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 8px;
            ">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Export Results
            </h2>
            <div class="export-content">
        """, unsafe_allow_html=True)
        
        # Add container styling for export section
        st.markdown("""
        <style>
        .export-content {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 1.5rem;
        }
        .stButton > button {
            background: linear-gradient(145deg, #ff4b4b, #ff6b6b) !important;
            color: white !important;
            font-weight: 500 !important;
            padding: 0.75rem 1.5rem !important;
            border: none !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            margin: 0.5rem 0 !important;
            box-shadow: 0 2px 4px rgba(255, 75, 75, 0.1) !important;
        }
        .stButton > button:hover {
            background: linear-gradient(145deg, #ff6b6b, #ff4b4b) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(255, 75, 75, 0.25) !important;
            color: white !important;
        }
        .export-button {
            background: linear-gradient(145deg, #ff4b4b, #ff6b6b);
            color: white !important;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            width: 100%;
            max-width: 300px;
            margin: 0.75rem 0;
            border: none;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(255, 75, 75, 0.1);
        }
        .export-button:hover {
            background: linear-gradient(145deg, #ff6b6b, #ff4b4b);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(255, 75, 75, 0.25);
            color: white !important;
            text-decoration: none;
        }
        .export-button svg {
            stroke: white !important;
        }
        .success-message {
            background-color: rgba(40, 167, 69, 0.1);
            border: 1px solid rgba(40, 167, 69, 0.2);
            color: #28a745;
            padding: 12px 16px;
            border-radius: 8px;
            margin-top: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)

        # Create columns for better layout
        export_col1, export_col2, export_col3 = st.columns([1, 2, 1])
        
        with export_col2:
            # Excel Export Button
            if st.button('Generate Excel Report', key='export_button', use_container_width=True):
                try:
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
                    
                    # Generate download link with better styling
                    st.markdown(f"""
                        <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" 
                           download="tool_wear_predictions.xlsx"
                           class="export-button">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                <polyline points="7 10 12 15 17 10"/>
                                <line x1="12" y1="15" x2="12" y2="3"/>
                            </svg>
                            Download Excel Report
                        </a>
                        """, unsafe_allow_html=True)
                    
                    # Show success message with icon
                    st.markdown("""
                        <div class="success-message">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                                <polyline points="22 4 12 14.01 9 11.01"/>
                            </svg>
                            <span>Excel report generated successfully!</span>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred while generating the Excel file: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Close the container div
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
