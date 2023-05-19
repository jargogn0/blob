import streamlit as st
st.set_page_config(page_title="Jargoñ Neural Network", page_icon=":spider:", layout="wide")
st.title("Hydrological :green[Modelling] :bar_chart:")
st.subheader("Developed by :red[Jargoñ] :spider:")
import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import seaborn as sns
import os
import base64
import io
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import plot_confusion_matrix
#from sklearn.metrics import plot_roc_curve
#import pandas_bokeh
from streamlit_lottie import st_lottie 
import requests



def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code!= 200:
        return None
    return r.json()

lottie_coding=load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qsSzatuXqF.json")

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("About us")
        st.write("##")
        st.write(
            """
            This is Hydrologial  Modelling "

            If this sounds interesting to you, consider subscribing and turning on the notifications, so you don’t miss any content.
            """
        )
        st.write("For more information about Jargoñ go to our [Jargoñ](https://www.jargoñ.com)")
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()

st.sidebar.info(
    """
    - Web App URL: <https:www.jargoñ.com/>
    - GitHub repository: <hhttps://jargogn0-blob-blob-twnpf9.streamlit.app/>
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Jargoñ: <https://jargoñ.com>
    [GitHub](https://github.com/jargon0)
    """
)

def main():
    st.subheader("Model 2: Predict Runoff :spider:")
    st.write("This application accepts a dataset with hydrological data, performs data cleaning and exploration, and builds several predictive models to forecast the 'Discharge' variable.")

     # Ask the user for the catchment name and area
    catchment_name = st.text_input('Enter the catchment name:')
    catchment_area = st.number_input('Enter the catchment area in Km2:', min_value=0.0, step=0.1)

    if catchment_name and catchment_area:
        st.write(f'You entered catchment name: {catchment_name} and catchment area: {catchment_area} Km2')

    st.subheader("Upload your data")

    uploaded_file_train = st.file_uploader("Choose a training file (optional)", type=["csv", "xlsx", "txt"])
    uploaded_file_val = st.file_uploader("Choose a validation file", type=["csv", "xlsx", "txt"])

    if uploaded_file_train is not None:
        # If the user uploads their own training data
        if uploaded_file_train.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file_train, delimiter=',')
        elif uploaded_file_train.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file_train)
        elif uploaded_file_train.name.endswith('.txt'):
            data = pd.read_csv(uploaded_file_train, delimiter='\t')
    else:
        # If the user does not upload their own training data, use your training data
        # If the user does not upload their own training data, use the default training data
        data = pd.read_excel('sn_river_data.xlsx')
        data_copy=data.copy()

    if uploaded_file_val is not None:
        # If the user uploads their own validation data
        if uploaded_file_val.name.endswith('.csv'):
            data_val = pd.read_csv(uploaded_file_val, delimiter=',')
        elif uploaded_file_val.name.endswith('.xlsx'):
            data_val = pd.read_excel(uploaded_file_val)
        elif uploaded_file_val.name.endswith('.txt'):
            data_val = pd.read_csv(uploaded_file_val, delimiter='\t')


    # Add a button to run the model
    run_model = st.button("Run Model 2")

    if run_model:

        def clean_data(data):
            data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
            data = data.replace(',', '.', regex=True)
            for col in data.columns:
                try:
                    data[col] = data[col].astype(float)
                except Exception as e:
                    print(f"Couldn't convert column {col} to float. Error: {e}")
            data.set_index('Date', inplace=True)

            if data.isnull().sum().sum() > 0:
                imputer = SimpleImputer(strategy='median')
                data_imputed = imputer.fit_transform(data)
                data = pd.DataFrame(data_imputed, columns=data.columns)

            return data


        # Convert date to datetime
        data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
        # Replace ',' with '.' for decimal notation
        data = data.replace(',', '.', regex=True)

        # Convert discharge from m³/s to mm/day
        data['Runoff (mm)'] = data['Discharge'] * 86400 / (catchment_area * 10**6)

        # For each column, try to convert it to float and catch exceptions
        for col in data.columns:
            try:
                data[col] = data[col].astype(float)
            except Exception as e:
                print(f"Couldn't convert column {col} to float. Error: {e}")
        
    

        # set index to 'Date'
        data.set_index('Date', inplace=True)

        #Description of the dataset
        desc = data.describe()
        st.write(desc.T)

        # name of the target column
        col_target = 'Discharge'
        # define the list of possible predictors 
        cols_pred = list(data.drop(columns = col_target).columns)

        # Handle missing data
        if data.isnull().sum().sum() > 0:
            st.write('There are missing values in the dataset, applying median imputation...')
            # Perform median imputation
            imputer = SimpleImputer(strategy='median')
            data_imputed = imputer.fit_transform(data)
            data = pd.DataFrame(data_imputed, columns=data.columns)

        #scatter bokeh
        #scatter_fig= data.plot_bokeh.scatter("Rainfall", "Discharge")
        #st.bokeh_chart(scatter_fig)

                   # Create two columns
            col1, col2 = st.beta_columns(2)

            # Box plot
            col1.subheader('Box Plot of Discharge')
            fig1 = plt.figure(figsize=(8, 6))
            sns.boxplot(y=data['Discharge'], palette='viridis')
            sns.despine()
            col1.pyplot(fig1)

            # Area plot
            col2.subheader('Area Plot of Discharge Over Time')
            fig2 = plt.figure(figsize=(8, 6))
            plt.fill_between(data.index, data['Discharge'], color='mediumseagreen', alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('Discharge')
            sns.despine()
            col2.pyplot(fig2)

            # Correlation heatmap (if you have many numerical variables)
            col1.subheader('Heatmap of Variable Correlations')
            fig3 = plt.figure(figsize=(8, 6))
            sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=1)
            col1.pyplot(fig3)

            # Line Plot of ETP, Rainfall, and Discharge Over Time
            col2.subheader('Line Plot of ETP, Rainfall, and Discharge Over Time')
            fig4, ax = plt.subplots(figsize=(8, 6))
            ax.plot(data.index, data['Rainfall'], label='Rainfall', color='steelblue')
            ax.plot(data.index, data['Runoff (mm)'], label='Runoff', color='darkorange')
            ax.legend()
            sns.despine()
            col2.pyplot(fig4)

            # Refresh columns
            col1, col2 = st.beta_columns(2)

            # Area Plot of ETP, Rainfall, and Discharge Over Time
            col1.subheader('Area Plot of ETP, Rainfall, and Discharge Over Time')
            fig5, ax = plt.subplots(figsize=(8, 6))
            #ax.fill_between(data.index, data['ETP'], label='ETP', color='firebrick', alpha=0.5)
            ax.fill_between(data.index, data['Rainfall'], label='Rainfall', color='steelblue', alpha=0.5)
            ax.fill_between(data.index, data['Runoff (mm)'], label='Runoff', color='darkorange', alpha=0.5)
            ax.legend()
            sns.despine()
            col1.pyplot(fig5)

            # Continue this way for the rest of your plots...

            # Refresh columns
            col1, col2 = st.beta_columns(2)

            # Display data
            col1.write(data_copy.head())
            col2.write(data.head())

            if 'Date' in data_copy.columns:
                data['Year'] = pd.to_datetime(data_copy['Date']).dt.year

            # Plot calculated vs USGS annual max flow values
            col1.subheader('Annual Maxima - USGS Peak Flow vs Daily Calculated')
            fig6, ax = plt.subplots(figsize=(6, 4))
            ax.plot(data['Year'], data['Runoff (mm)'], color="purple", linestyle=':', marker='o', label="USGS Annual Max")
            ax.plot(data['Year'], data['Runoff (mm)'], color="lightgrey", linestyle=':', marker='o', label="Calculated Annual Max")
            ax.legend()
            ax.set_title("Annual Maxima - USGS Peak Flow vs Daily Calculated")
            sns.despine()
            col1.pyplot(fig6)

            # Plot calculated vs USGS annual max flow values for Rainfall
            col2.subheader('Annual Maxima - USGS Peak Flow vs Daily Calculated for Rainfall')
            fig7, ax = plt.subplots(figsize=(6, 4))
            ax.plot(data['Year'], data['Rainfall'], color="purple", linestyle=':', marker='o', label="USGS Annual Max")
            ax.plot(data['Year'], data['Rainfall'], color="lightgrey", linestyle=':', marker='o', label="Calculated Annual Max")
            ax.legend()
            ax.set_title("Annual Maxima - USGS Peak Flow vs Daily Calculated")
            sns.despine()
            col2.pyplot(fig7)

            # Print column names
            col1.write(data_copy.columns)

            # Plot your data
            col2.subheader('Cumulative Sum & Daily Mean Discharge')
            fig8, ax = plt.subplots(figsize=(8, 6))
            data['Runoff (mm)'].plot(ax=ax, label = "Cumulative Volume")

            # Make the y-axis label, ticks and tick labels match the line color.
            ax.set_ylabel('Total Area Runoff', color='b')
            ax.tick_params('y', colors='b')

            ax2 = ax.twinx()
            ax2.scatter(x=data['Year'], y=data['Discharge'], marker="o", s=4, color ="purple", label="Daily Mean")
            ax2.set_ylabel('Stream Discharge (CFS)', color='purple')
            ax2.tick_params('y', colors='purple')
            ax2.set_ylim(0,500)
            ax.set_title("Cumulative Sum & Daily Mean Discharge")
            ax.legend()
            # Reposition the second legend so it renders under the first legend item
            ax2.legend(loc = "upper left", bbox_to_anchor=(0.0, 0.5))
            fig8.tight_layout()
            col2.pyplot(fig8)

        # Split data into train and validation
        Xtr, Xval, Ytr, Yval = train_test_split(data[cols_pred], data[col_target], test_size=0.2, random_state=42)

        # Dictionary of models
        models = {'Linear Regression': LinearRegression(),
                  'Random Forest': RandomForestRegressor(random_state=42),
                  'AdaBoost': AdaBoostRegressor(random_state=42),
                  'Decision Tree': DecisionTreeRegressor(random_state=42)}

        # Dictionary for model hyperparameters
        params = {'Random Forest': {'n_estimators': [100, 200, 300, 400, 500],
                                    'max_depth': [5, 10, 15, 20, 25, None],
                                    'min_samples_split': [2, 5, 10],
                                    'min_samples_leaf': [1, 2, 4],
                                    'bootstrap': [True, False]},
                  'AdaBoost': {'n_estimators': [50, 100, 150, 200, 250],
                               'learning_rate': [0.001, 0.01, 0.1, 1, 10]},
                  'Decision Tree': {'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                                    'splitter': ['best', 'random'],
                                    'max_depth': [5, 10, 15, 20, 25, None],
                                    'min_samples_split': [2, 5, 10],
                                    'min_samples_leaf': [1, 2, 4]}}

        # Fit and tune the models
        models_best = {}
        for name, model in models.items():
            if name in params:
                model_cv = RandomizedSearchCV(model, params[name], cv=5, random_state=42, n_jobs=-1)
                model_cv.fit(Xtr, Ytr)
                model = model_cv.best_estimator_
            else:
                model.fit(Xtr, Ytr)
            models_best[name] = model

        # After training and evaluating the models
        st.write("The Root Mean Square Error (RMSE) displayed for each model gives an indication of the model's performance - the lower the RMSE, the better the model's performance. These are the RMSE values for our models:")
        model_names = []
        model_rmse = []
        for name, model in models_best.items():
            Ytr_pred = model.predict(Xtr)
            Yts_pred = model.predict(Xval)
           
            rmse_tr = np.sqrt(mean_squared_error(Ytr, Ytr_pred))
            rmse_val = np.sqrt(mean_squared_error(Yval, Yts_pred))
            model_names.append(name)
            model_rmse.append(rmse_val)
            st.write(f"Model: {name}, RMSE on training data: {rmse_tr}, RMSE on validation data: {rmse_val}")

            # Bar plot to compare RMSE of all models
            fig, ax = plt.subplots()
            ax.bar(model_names, model_rmse)
            ax.set_xlabel("Models")
            ax.set_ylabel("RMSE")
            ax.set_title("RMSE Comparison Between Models")
            st.pyplot(fig)


        # Select the best model based on RMSE on validation data
        best_model_name = model_names[np.argmin(model_rmse)]
        best_model = models_best[best_model_name]
        st.write(f"The best model is {best_model_name} with RMSE {np.min(model_rmse)} on the validation data")

        # Interpreting the model's performance
        RMSE_THRESHOLD = 10  # This is just an example, please adjust according to your data and domain knowledge
        if np.min(model_rmse) < RMSE_THRESHOLD:
            st.write(f"The model's performance is satisfactory as the RMSE on validation data is less than {RMSE_THRESHOLD}")
        else:
            st.write(f"The model's performance is not satisfactory as the RMSE on validation data is more than {RMSE_THRESHOLD}")

        # Use the best model to make predictions on the validation data
        Yval_pred = best_model.predict(Xval)

        # Add predictions to the validation data
        val_data = Xval.copy()
        val_data[col_target] = Yval
        val_data['Predicted_Discharge'] = Yval_pred

        # Create download link for the validation data with filled missing values and predictions
        st.markdown(get_table_download_link(val_data, "validation_data_with_predictions.csv"), unsafe_allow_html=True)

       # Adding all model's predictions to the validation data
        val_data['Linear Regression_predictions'] = models_best['Linear Regression'].predict(Xval)
        val_data['Random Forest_predictions'] = models_best['Random Forest'].predict(Xval)
        val_data['AdaBoost_predictions'] = models_best['AdaBoost'].predict(Xval)
        val_data['Decision Tree_predictions'] = models_best['Decision Tree'].predict(Xval)

        # Create download link for the validation data with filled missing values and predictions from all models
        st.markdown(get_table_download_link(val_data, "validation_data_with_all_predictions.xlsx"), unsafe_allow_html=True)



def get_table_download_link(df, filename):
    # function to create a link to download a dataframe as an excel file
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, header=True) 
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()  
    href = f'<a href="data:file/xlsx;base64,{b64}" download={filename}>Download Excel file</a>'
    return href


st.subheader('Model 1 - Predicting Next Rainfall and Flood Risk')

uploaded_file_2 = st.file_uploader("Choose a new TRAINING csv or Excel file for MODEL 1", type=['csv', 'xlsx'])

if uploaded_file_2 is not None:
    try:
        data = pd.read_csv(uploaded_file_2)
    except Exception as e:
        data = pd.read_excel(uploaded_file_2)

    st.write("Training data for Model 1:")
    st.write(data.head())

else:
    data = pd.read_excel('rain_data.xlsx')
    st.write("Using default training data for Model 1:")
    st.write(data.head())

# Upload the data for Model 2
file_upload_2 = st.file_uploader("Upload data for Model 1", type=["csv", "xlsx"])

if file_upload_2:
    if file_upload_2.type == "text/csv":
        data = pd.read_csv(file_upload_2)
    elif file_upload_2.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        data = pd.read_excel(file_upload_2)
    else:
        st.write("Please upload a valid file format.")

# Add a button to run Model 1
if 'run_model_1' not in st.session_state:
    st.session_state['run_model_1'] = False

if st.button("Run Model 1"):
    st.session_state.run_model_1 = True

if st.session_state.run_model_1 and file_upload_2:
    # Define input features and target variable
    features = ['Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmph', 'Elevation_m', 'Population_Density', 'Flood_History_Count']
    target = 'Flood_Status'

    # Replace non-numeric values in numeric columns
    numeric_cols = ['Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmph', 'Elevation_m', 'Population_Density', 'Flood_History_Count']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with NaN values
    data = data.dropna()

    st.line_chart(data.Rainfall_mm)
    st.line_chart(data.Temperature_C)

    # Prepare the dataset, excluding the 'Timestamp' column
    X = data[features].copy()
    X = X.drop(['Timestamp'], errors='ignore', axis=1)

    y = data[target]

    # Encode the target variable (Flood_Status) as binary (0: No, 1: Yes)
    y = y.map({'No': 0, 'Yes': 1})
    st.write("Value counts for target variable:")
    st.write(y.value_counts())


    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Standardize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Add session state for model selection
    if 'model_selection' not in st.session_state:
        st.session_state['model_selection'] = 'Random Forest'
    model_options = ('Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'K-Nearest Neighbors')
    model_selection = st.selectbox(
        'Select a model:',
        model_options,
    index=model_options.index(st.session_state['model_selection'])
    )
    st.session_state['model_selection'] = model_selection

    # Train the selected model
    if model_selection == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_selection == 'Logistic Regression':
        clf = LogisticRegression(random_state=42)
    elif model_selection == 'Support Vector Machine':
        clf = SVC(random_state=42)
    elif model_selection == 'Decision Tree':
        clf = DecisionTreeClassifier(random_state=42)
    elif model_selection == 'K-Nearest Neighbors':
        clf = KNeighborsClassifier()

    clf.fit(X_train, y_train)

    # Validate the model
    y_val_pred = clf.predict(X_val)
    st.write("Validation set results:")
    st.write(classification_report(y_val, y_val_pred))
    st.write(confusion_matrix(y_val, y_val_pred))

    # Test the model
    y_test_pred = clf.predict(X_test)
    st.write("Test set results:")
    st.write(classification_report(y_test, y_test_pred))
    st.write(confusion_matrix(y_test, y_test_pred))

    # Calculate and display metrics
    st.write("Model 2 Performance Metrics:")
    st.write(classification_report(y_test, y_test_pred))

    # Calculate and display feature importances
    if hasattr(clf, 'feature_importances_'):
        feature_importances = pd.DataFrame(clf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)
        st.write("Model 2 Feature Importances:")
        st.write(feature_importances)

    # Generate a download link for the prediction results
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    
    
    import altair as alt

    #data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    line_chart = alt.Chart(data).mark_line().encode(
        x='Timestamp:T',
        y='Rainfall_mm:Q',
        tooltip=['Timestamp', 'Rainfall_mm']
    ).interactive().properties(
        title='Rainfall over Time'
    )

    st.altair_chart(line_chart, use_container_width=True)


    if st.session_state.run_model_1 and file_upload_2:
        # Plot confusion matrix
        #fig, ax = plt.subplots()
        #plot_confusion_matrix(clf, X_test, y_test, ax=ax, cmap='Blues')
        #ax.set_title(f"Model 1 ({st.session_state['model_selection']}) - Confusion Matrix")
        #st.pyplot(fig)

        # Plot ROC curve
        #fig, ax = plt.subplots()
        #plot_roc_curve(clf, X_test, y_test, ax=ax)
        #ax.set_title(f"Model 1 ({st.session_state['model_selection']}) - ROC Curve")
        #st.pyplot(fig)

    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="model_predictions.xlsx">Download MODEL Predictions Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)


else:
    st.write("Please upload data before running Model 1.")

if __name__ == "__main__":
    main()




     


