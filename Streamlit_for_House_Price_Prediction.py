import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)
#==============================================================================================================================================================================
st.image(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\Website Pics\Gemini_Generated_Image_j3ppk5j3ppk5j3pp.png",use_container_width="stretch",width=3000)
st.subheader("Welcome to the AI-powered Property Price Prediction App! " \
            "This application allows you to input various features of a property and receive an estimated price based on our trained machine learning model.",
            icon=":material/apartment:",divider="grey")
st.caption('Get accurate house value forecasts, investment insights, and 5-year price predictions powered by Machine Learning.',text_alignment='left')
st.sidebar.subheader("AI Prediction",divider="grey",icon=":material/apartment:")
# Use below link to use icons
# icon=":material/apartment:" This wwas got from below website
# https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded&icon.query=house&selected=Material+Symbols+Rounded:apartment:FILL@0;wght@400;GRAD@0;opsz@24&icon.size=24&icon.color=%23e3e3e3

#==============================================================================================================================================================================

# Below syntax is given by Ai which is looking good
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        "Future Value Prediction",
        ["Home", "Predict", "Contact Us"],
        icons=["house-fill","globe", "people"],
        default_index=0
    )

# Below are the icons available in icons package
# house,house-fill,person,person-fill,people,people-fill,search,gear,gear-fill,table,database,
# calendar,calendar-event,trophy,award,star,star-fill,heart,heart-fill
# book,clipboard-data,bar-chart,bar-chart-fill,graph-up,graph-up-arrow,
# graph-down,pie-chart,globe,camera,image,file-earmark,folder,download,
# upload,play-circle,pause-circle,check-circle,x-circle,info-circl

#==============================================================================================================================================================================

if selected == "Home":

    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #0f172a, #1e293b);
            padding:25px;
            border-radius:15px;
            border:1px solid rgba(255,255,255,0.1);
            margin-bottom:20px;
        ">
    
        <h2 style="color:white; margin-bottom:10px;">
        🏡 AI-Powered Property Price Prediction
        </h2>
    
        <p style="color:#cbd5e1; font-size:16px;">
        Our platform helps buyers, investors, and real estate professionals
        predict property values, evaluate investment opportunities,
        and forecast future prices with advanced Machine Learning models.
        </p>
    
        <hr style="border-color:#334155;"/>
    
        <h4 style="color:#60a5fa;">📊 Prediction Output</h4>
    
        <ul style="color:#e2e8f0;">
        <li>✅ House Price Prediction</li>
        <li>✅ Future Value Forecasting </li>
        <li>✅ Investment Analysis</li>
        <li>✅ Market Insights & Trends</li>
        </ul>
    
        </div>
        """, unsafe_allow_html=True)

#==============================================================================================================================================================================
    
if selected == "Predict":
    # st.title("🏡 Property Price Prediction")

    # st.caption(
    #     "Enter property details to estimate the current market value, "
    #     "future price appreciation, and investment potential."
    # )
    # st.info(
    # "Fill in the property details below and click Predict to receive "
    # "AI-powered price estimations and investment insights."
    # )

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding:25px;
        border-radius:15px;
        border:1px solid rgba(255,255,255,0.1);
        margin-bottom:20px;
    ">

    <h2 style="color:white; margin-bottom:10px;">
    🏡 Property Price Prediction
    </h2>

    <p style="color:#cbd5e1; font-size:16px;">
    Enter property details below to estimate the current market value,
    future property price, and investment potential using AI-powered
    machine learning models.
    </p>

    <hr style="border-color:#334155;"/>

    <ul style="color:#e2e8f0;">

    </ul>

    </div>
    """, unsafe_allow_html=True)

    #===============================================================
    col1, col2, col3 = st.columns(3)

    #===============================================================

    col4, col5, col6 = st.columns(3)

    with col4:
        
        State = st.selectbox(
                    "State",
                    ['Andhra Pradesh','Assam','Bihar','Chhattisgarh','Delhi','Gujarat',
                    'Haryana','Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra',
                    'Odisha','Punjab','Rajasthan','Tamil Nadu','Telangana','Uttar Pradesh',
                    'Uttarakhand','West Bengal'])
        City = st.selectbox(
                    "City",
                    ['Ahmedabad', 'Amritsar', 'Bangalore', 'Bhopal', 'Bhubaneswar', 'Bilaspur',
                    'Chennai', 'Coimbatore', 'Cuttack', 'Dehradun', 'Durgapur', 'Dwarka',
                    'Faridabad', 'Gaya', 'Gurgaon', 'Guwahati', 'Haridwar', 'Hyderabad', 'Indore',
                    'Jaipur', 'Jamshedpur', 'Jodhpur', 'Kochi', 'Kolkata', 'Lucknow', 'Ludhiana',
                    'Mangalore', 'Mumbai', 'Mysore', 'Nagpur', 'New Delhi', 'Noida', 'Patna', 'Pune',
                    'Raipur', 'Ranchi', 'Silchar', 'Surat', 'Trivandrum', 'Vijayawada',
                    'Vishakhapatnam', 'Warangal'])
        Locality = st.selectbox(
                    "Locality",
                    ['Locality_1', 'Locality_10', 'Locality_100', 'Locality_101', 'Locality_102',
                    'Locality_103', 'Locality_104', 'Locality_105', 'Locality_106',
                    'Locality_107', 'Locality_108', 'Locality_109', 'Locality_11', 'Locality_110',
                    'Locality_111', 'Locality_112', 'Locality_113', 'Locality_114',
                    'Locality_115', 'Locality_116', 'Locality_117', 'Locality_118',
                    'Locality_119', 'Locality_12', 'Locality_120', 'Locality_121', 'Locality_122',
                    'Locality_123', 'Locality_124', 'Locality_125', 'Locality_126',
                    'Locality_127', 'Locality_128', 'Locality_129', 'Locality_13', 'Locality_130',
                    'Locality_131', 'Locality_132', 'Locality_133', 'Locality_134',
                    'Locality_135', 'Locality_136', 'Locality_137', 'Locality_138',
                    'Locality_139', 'Locality_14', 'Locality_140' ,'Locality_141', 'Locality_142',
                    'Locality_143', 'Locality_144', 'Locality_145', 'Locality_146',
                    'Locality_147', 'Locality_148', 'Locality_149', 'Locality_15', 'Locality_150',
                    'Locality_151', 'Locality_152', 'Locality_153', 'Locality_154'])
        Property_Type = st.selectbox(
                    "Property_Type",
                    ['Apartment', 'Independent House', 'Villa'])
        
        BHK = st.number_input("BHK", 1, 5)
        Size_in_SqFt = st.number_input("Area (Sq Ft)",500)

        Price_per_SqFt = st.number_input("Price_per_SqFt",1)
        price_from_user = Price_per_SqFt/100000 # Converting the Price to Model trained way

        Floor_No = st.number_input("Floor_No",1,30)


    with col5:

        Total_Floors = st.number_input("Total_Floors",1,30)

        # Below one is encode by lable encoder
        Security = st.selectbox( "Security",["Yes","No"])

        # Below one is encode by lable encoder
        Facing = st.selectbox("Facing",['East', 'North', 'South', 'West']) 

        # Below one is encode by lable encoder
        Owner_Type = st.selectbox("Owner_Type",['Broker', 'Builder', 'Owner'])

        parking = st.selectbox("Parking Spaces", ["Yes", "No"])
        parking_map = {"No": 0,"Yes": 1}
        parking_encoded = parking_map[parking]

        Transport_Accessibility = st.selectbox("Transport_Accessibility",['Low','Medium','High'])
        transport_map = {'Low': 1,'Medium': 2,'High': 3}
        transport_encoded = transport_map[Transport_Accessibility]

        Furnished_Status = st.selectbox("Furnished Status",['Furnished','Unfurnished'])  
        furnishing_map = {'Unfurnished': 1,'Furnished': 2}
        furnishing_encoded = furnishing_map[Furnished_Status]

        Availability_Status = st.selectbox("Availability_Status",['Ready_to_Move', 'Under_Construction'])
        Availability_map = {'Under_Construction': 1,'Ready_to_Move': 2}
        Availability_encoded = Availability_map[Availability_Status]


    with col6:

        Has_Gym = st.selectbox("Gym",["Yes","No"])
        Gym_map = {"No": 0,"Yes": 1}
        Gym_encoded = Gym_map[Has_Gym]

        Has_Playground = st.selectbox("Playground",["Yes","No"])
        Playground_map = {"No": 0,"Yes": 1}
        Playground_encoded = Playground_map[Has_Playground]

        Has_Garden = st.selectbox("Garden",["Yes","No"])
        Garden_map = {"No": 0,"Yes": 1}
        Garden_encoded = Garden_map[Has_Garden]

        Has_Clubhouse = st.selectbox("Clubhouse",["Yes","No"])
        Clubhouse_map = {"No": 0,"Yes": 1}
        Clubhouse_encoded = Clubhouse_map[Has_Clubhouse]

        Has_Pool = st.selectbox("Pool",["Yes","No"])
        Pool_map = {"No": 0,"Yes": 1}
        Pool_encoded = Pool_map[Has_Pool]

        current_year = datetime.datetime.now().year
        Year_Built = st.slider('Year_Built',1990,current_year)
        Age_of_Property = (current_year-Year_Built)

        Nearby_Schools = st.slider('Nearby_Schools',1,10)
        
        Nearby_Hospitals = st.slider('Nearby_Hospitals',1,10)
        
        Predict_the_price_using = st.selectbox("Select the Model to Predict Price",['LR','Decission Tree','Random Forest','XGBoost','LightGBM'])


    user_data = {'State':State,
                'City':City,
                'Locality':Locality,
                'Property_Type':Property_Type,
                'BHK':BHK,
                'Size_in_SqFt':Size_in_SqFt,
                'Price_per_SqFt':price_from_user,
                'Floor_No':Floor_No,
                'Total_Floors':Total_Floors,
                'Age_of_Property':Age_of_Property,
                'Nearby_Schools':Nearby_Schools,
                'Nearby_Hospitals':Nearby_Hospitals,
                'Security':Security,
                'Facing':Facing,
                'Owner_Type':Owner_Type,
                'Parking_Score':parking_encoded,
                'Transport_Score':transport_encoded,
                'Furnishing_Score':furnishing_encoded,
                'Availability_Score':Availability_encoded,
                'Has_Gym':Gym_encoded,
                'Has_Playground':Playground_encoded,
                'Has_Garden':Garden_encoded,
                'Has_Clubhouse':Clubhouse_encoded,
                'Has_Pool':Pool_encoded}

    #====================================================================================================

    # Loading Regression models
    LR_model = joblib.load(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\Trained Models\LR_Model.pkl")
    RF_model = joblib.load(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\Trained Models\Random_Forest.pkl")
    DT_model  = joblib.load(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\Trained Models\Decission_Tree.pkl")
    XG_model = joblib.load(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\Trained Models\XG_Booster.pkl")
    LGBM_model = joblib.load(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\Trained Models\LightGBM.pkl")

    # Loading encoders, scaler and features
    encoders = joblib.load(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\Trained Models\encoders.pkl")
    scaler = joblib.load(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\Trained Models\scaler.pkl")
    features = joblib.load(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\Trained Models\features.pkl")

    

    input_df = pd.DataFrame([user_data]) # Creating the Data frame from User input received from website
    input_df = input_df.reindex(columns=features).copy() # Re-arranging the features as per model trained way

    if st.button("🚀 Predict Property Value", use_container_width=300):

        if Predict_the_price_using == 'LR':
        #==================Load LR Model=====================
            # Encode categorical columns
            for col in encoders.keys():
                if col in input_df.columns:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str) )

            scaled_inputs = scaler.transform(input_df)

            prediction = LR_model.predict(scaled_inputs)
            predicted_price = prediction[0]

        #==================Load Decission Tree Model=====================
        elif Predict_the_price_using == 'Decission Tree':

            # Encode categorical columns
            for col in encoders.keys():
                if col in input_df.columns:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str) )

            scaled_inputs = scaler.transform(input_df)

            prediction = DT_model.predict(scaled_inputs)
            predicted_price = prediction[0]

        #==================Load Random forest Model=====================    
        elif Predict_the_price_using == 'Random Forest':
            # Encode categorical columns
            for col in encoders.keys():
                if col in input_df.columns:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str) )

            scaled_inputs = scaler.transform(input_df)

            prediction = RF_model.predict(scaled_inputs)
            predicted_price = prediction[0]

        #==================Load XGBoost Model===================== 
        elif Predict_the_price_using == 'XGBoost':

            # Encode categorical columns
            for col in encoders.keys():
                if col in input_df.columns:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str) )

            scaled_inputs = scaler.transform(input_df)

            prediction = XG_model.predict(scaled_inputs)
            predicted_price = prediction[0]

        #==================Load LightGBM Model=====================
        elif Predict_the_price_using == 'LightGBM':

            # Encode categorical columns
            for col in encoders.keys():
                if col in input_df.columns:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str) )

            scaled_inputs = scaler.transform(input_df)

            prediction = LGBM_model.predict(scaled_inputs)
            predicted_price = prediction[0]


        
        st.write("In lakhs:- ",round(predicted_price,2))

        current_price = (price_from_user*Size_in_SqFt)
        with col1:
            st.metric("Current Price", f"₹ {round(current_price,2)} Lakhs")

        with col2:
            st.metric("Future Value (5 Years)", f"₹{round(predicted_price,2)} Lakhs")

        with col3:
            percentage_increase = ((round(predicted_price,2) - current_price) / current_price) * 100
            st.metric("Growth", f"{round(percentage_increase,2)} %")

        st.write("The option which you have selected are",user_data)