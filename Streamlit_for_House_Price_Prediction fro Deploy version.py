import streamlit as st
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)
#==============================================================================================================================================================================
st.image("Gemini_Generated_Image.png",use_container_width="stretch",width=3000)
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
        ["Home","Investment Analysis", "Predict","Metrix Score", "Contact Us"],
        icons=["house-fill","bar-chart","globe", "table","people"],
        default_index=0
    )

# Below are the icons available in icons package
#---------------------------------------------------
# house,house-fill,person,person-fill,people,people-fill,search,gear,gear-fill,table,database,
# calendar,calendar-event,trophy,award,star,star-fill,heart,heart-fill
# book,clipboard-data,bar-chart,bar-chart-fill,graph-up,graph-up-arrow,
# graph-down,pie-chart,globe,camera,image,file-earmark,folder,download,
# upload,play-circle,pause-circle,check-circle,x-circle,info-circl



#==============================================================================================================================================================================
#----------------------------------------------------------------------------Home Page Design----------------------------------------------------------------------------
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

#==============================================================================================================================================================================
#----------------------------------------------------------------------------Investment Analysis----------------------------------------------------------------------------
#==============================================================================================================================================================================

if selected == "Investment Analysis":
    # Processed data by me
    df = pd.read_parquet("housing_data.parquet")
    
    df = df.drop(['Unnamed: 0','ID'],axis=1) # Dropping the index and ID
    # Original Data by Guvi
    # df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Ai-ML Data Science Guvi\Projects\Project 3 ML for Real estate\india_housing_prices Data.csv")
    st.write(df.shape)

    Properties_col,Price_col,SqFt_col,BHK_col = st.columns(4)
    

    st.sidebar.header("Filters")

    selected_city = st.sidebar.selectbox(
        "Select City",["All"] + list(df["City"].unique()))

    if selected_city != "All":
        filtered_df = df[df["City"] == selected_city]
    else:
        filtered_df = df



    selected_BHK = st.sidebar.selectbox(
        "BHK",["All"] + list(filtered_df["BHK"].unique()))

    if selected_BHK != "All":
        filtered_BHK_df = filtered_df[filtered_df["BHK"] == selected_BHK]
    else:
        filtered_BHK_df = filtered_df



    st.subheader(f"The Details are showing for {selected_city}",text_alignment="center",divider=True)

    Properties_col.metric("Total Properties",len(filtered_BHK_df))
    Price_col.metric("Average Price in Lakhs",round(filtered_BHK_df["Price_in_Lakhs"].mean(),2))
    SqFt_col.metric("Average SqFt",round(filtered_BHK_df["Size_in_SqFt"].mean(),2))
    BHK_col.metric("Average BHK",round(filtered_BHK_df["BHK"].mean(),2))

    bar1,bar2 = st.columns(2)
    bar3,bar4 = st.columns(2)
    bar5,bar6 = st.columns(2)

    # 0. Price Distribution
    st.subheader(f"Properties available at each city for {selected_city} BHK",text_alignment="center",divider="grey")
    fig, ax = plt.subplots(figsize=(10,3))
    colors = sns.color_palette("viridis", 30) # Using colors
    sns.countplot(
        data=filtered_BHK_df,
        x="City",
        palette=colors,)
    plt.xticks(rotation=90)
    for container in ax.containers:
        ax.bar_label(container,fontsize=5)
    st.pyplot(fig)

    # 1. Price Distribution
    st.subheader(f"Property Price Distribution for {selected_BHK} BHK",text_alignment="center",divider="grey")
    fig, ax = plt.subplots(figsize=(10,3))

    if selected_BHK != "All":
        filtered_BHK_df1 = df[df["BHK"] == selected_BHK]
    else:
        filtered_BHK_df1 = df
    colors = sns.color_palette("viridis", 30) # Using colors
    sns.histplot(
        data=filtered_BHK_df1,
        x="Price_in_Lakhs",
        bins=30, # How many Bars we want
        kde=True,
        palette=colors,
        ax=ax)
    for container in ax.containers:
        ax.bar_label(container,fontsize=5)
    st.pyplot(fig)

    with bar1:
        # 2. Grouping the Average price per city
        st.subheader(f"Average Price by City for {selected_BHK} BHK",text_alignment="center",divider="grey")
        city_price = (filtered_BHK_df1.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(10))
        fig, ax = plt.subplots(figsize=(5,3))
        colors = sns.color_palette("viridis", 10) # Using colors
        sns.barplot(
            x=city_price.values,
            y=city_price.index,
            palette=colors,
            ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f',fontsize=5)
        #plt.xticks(rotation=45) # Rotation is not required as x axis numbers 
        st.pyplot(fig)

    # 3. Price vs Property Size
    st.subheader(f"Price vs Property Size fro {selected_city} City {selected_city} BHK",text_alignment="center",divider="grey")
    fig, ax = plt.subplots(figsize=(10,3))

    if len(filtered_df) > 5000:
        plot_df = filtered_df.sample(5000, random_state=42)
    else:
        plot_df = filtered_df
    sns.scatterplot(
        #data=filtered_df.sample(5000), # When the filter was applied on Dataframe "filtered_df", if the rows are less then 5000 , we get error, hence if condition is used at above
        data=filtered_df.sample(3000),
        # data=plot_df, # Instead of using :Filtered df" use plot_df
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        alpha=0.4, # Maintain the Tranparnet 0 is Full transparent, 1 is Full Visible
        ax=ax)
    st.pyplot(fig)

    # 4. Heatmap (Must Have)
    st.subheader(f"Correlation Heatmap for {selected_city} City {selected_city} BHK",text_alignment="center",divider="grey")
    filtered_BHK_df = filtered_BHK_df.drop(['ID'],axis=1,errors='ignore') # Droping the ID column
    numeric_df = filtered_BHK_df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        annot_kws={"size": 5}, # Font size inside the heatmap
        fmt=".2f")
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    st.pyplot(fig)

    # 5. Property Type vs Price per SqFt
    with bar2:
        st.subheader(f"Property Type vs Price per SqFt for {selected_city} City {selected_city} BHK",text_alignment="center",divider="grey")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.boxplot(
            data=filtered_BHK_df,
            x="Property_Type",
            y="Price_per_SqFt")
        #plt.xticks(rotation=45)
        st.pyplot(fig)

    # 6. BHK Distribution
    with bar3:
        st.subheader(f"BHK Distribution for {selected_city} City",text_alignment="center",divider="grey")
        fig, ax = plt.subplots(figsize=(5,3))
        colors = sns.color_palette("viridis", 5) # Using colors
        sns.countplot(
            data=filtered_df,
            x="BHK",
            palette=colors)
        for container in ax.containers: # Adding Label on Top of the Bar
            ax.bar_label(container,fontsize=6)
        st.pyplot(fig)

    with bar4:
        # 7. Furnished Status vs Price
        st.subheader(f"Furnished Status vs Price {selected_city} City {selected_BHK} BHK",text_alignment="center",divider="grey")
        fig, ax = plt.subplots(figsize=(5,3))
        colors = sns.color_palette("viridis", 3) # Using colors
        sns.barplot(
            data=filtered_BHK_df,
            x="Furnished_Status",
            y="Price_in_Lakhs",
            palette=colors)
        for container in ax.containers:
            ax.bar_label(container,fontsize=6) # Adding Label on Top of the Bar
        st.pyplot(fig)

    # 8. Top Localities by Average Price
    with bar5:
        st.subheader(f"Top Localities by Average Price for {selected_city} City {selected_BHK} BHK",text_alignment="center",divider="grey")
        top_localities = (
        filtered_BHK_df.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(10))
        fig, ax = plt.subplots(figsize=(5,3))
        colors = sns.color_palette("viridis", len(top_localities)) # Using colors
        sns.barplot(
            x=top_localities.values,
            y=top_localities.index,
            palette=colors)
        for container in ax.containers: # Adding Label on Top of the Bar
            ax.bar_label(container,fontsize=6,padding=1,fmt='%.1f')
        st.pyplot(fig)

    # 9. Public Transport Availabilty
    with bar6:
        st.subheader(f"Top Localities by Average Price for {selected_city} City {selected_BHK} BHK",text_alignment="center",divider="grey")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.countplot(
            data=filtered_df,
            x="Public_Transport_Accessibility",
            order=["Low", "Medium", "High"],
            palette="viridis", # Colors
            ax=ax)
        
        for container in ax.containers:
            ax.bar_label(container,fontsize=6,padding=2)
        ax.set_title("Public Transport Accessibility")
        ax.set_xlabel("")
        ax.set_ylabel("Property Count")

        st.pyplot(fig)


#==============================================================================================================================================================================
#----------------------------------------------------------------------------Prediction Page Design----------------------------------------------------------------------------
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
        
        Predict_the_price_using = st.selectbox("Select the Model to Predict Price",['XGBoost','LightGBM'])


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
    #LR_model = joblib.load("LR_Model.pkl")
    #RF_model = joblib.load("Random_Forest.pkl")
    #DT_model  = joblib.load("Decission_Tree.pkl")
    XG_model = joblib.load("XG_Booster.pkl")
    LGBM_model = joblib.load("LightGBM.pkl")

    # Laoding Classification Models
    XG_model_Class =  joblib.load("xgboostClassification_model.pkl")
    LightGBM_model_Class =  joblib.load("lightgbmClassification_model.pkl")

    # Loading encoders, scaler and features
    encoders = joblib.load("encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")

    

    input_df = pd.DataFrame([user_data]) # Creating the Data frame from User input received from website
    input_df = input_df.reindex(columns=features).copy() # Re-arranging the features as per model trained way

    #===========Creating the Function to call inside the button=====================
    
    @st.dialog("🏠 Property Evaluation Result") # This line runs along with function
    def show_prediction():

        st.subheader("Predicted Property Value")

        st.metric("Estimated Price",
                    f"₹ {predicted_price:.2f} Lakhs")
        st.divider()

        col7, col8 = st.columns(2)

        with col7:
            st.metric("Current Price", f"₹ {round(current_price,2)} Lakhs")

        with col8:
            percentage_increase = ((round(predicted_price,2) - current_price) / current_price) * 100
            st.metric("Growth", f"{round(percentage_increase,2)} %")

        st.divider()
        if predicted_Class == 0:
            st.error("🤔 Poor Investment Opportunity")
        elif predicted_Class == 1:
            st.balloons()
            st.info("👏 Ok to Investment Opportunity")
        elif predicted_Class == 2:
            st.info("✅ Good Investment Opportunity")
        else:
            st.success("🌟 Excellent Investment Opportunity")

    #===============================================================================

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

            pred_results = XG_model_Class.predict(scaled_inputs)
            predicted_Class = pred_results[0]

        #==================Load LightGBM Model=====================
        elif Predict_the_price_using == 'LightGBM':

            # Encode categorical columns
            for col in encoders.keys():
                if col in input_df.columns:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str) )

            scaled_inputs = scaler.transform(input_df)

            prediction = LGBM_model.predict(scaled_inputs)
            predicted_price = prediction[0]

            pred_results = XG_model_Class.predict(scaled_inputs)
            predicted_Class = pred_results[0]


        
        st.write("In lakhs:- ",round(predicted_price,2))

        current_price = (price_from_user*Size_in_SqFt)
        with col1:
            st.metric("Current Price", f"₹ {round(current_price,2)} Lakhs")

        with col2:
            st.metric("Future Value (5 Years)", f"₹{round(predicted_price,2)} Lakhs")

        with col3:
            percentage_increase = ((round(predicted_price,2) - current_price) / current_price) * 100
            st.metric("Growth", f"{round(percentage_increase,2)} %")

        
        show_prediction()

        st.write("The option which you have selected are",user_data)

#==============================================================================================================================================================================
#----------------------------------------------------------------------------Metrix Page----------------------------------------------------------------------------
#==============================================================================================================================================================================
if selected == "Metrix Score":
    Regression_metrix_df = pd.read_csv("Regression models results.csv")
    classification_metrics_df  = pd.read_csv("Classfication model results.csv")

    Reg_cal,Class_cal = st.columns(2)

    with Reg_cal:
        st.subheader("Regression Metrics",text_alignment="center")
        st.dataframe(Regression_metrix_df)
    with Class_cal:
        st.subheader("Classification Metrics",text_alignment="center")
        st.dataframe(classification_metrics_df)

    
