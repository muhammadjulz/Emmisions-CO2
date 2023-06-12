import streamlit as st
import pandas as pd
import joblib


# Load files 

with open('pipeline_model_randomforest.pkl', 'rb') as file_1:
  pipeline_model_randomforest = joblib.load(file_1)



def run():
    st.markdown("<h1 style='text-align: center;'>EcoEmission Predictor!</h1>", unsafe_allow_html=True)
    st.subheader('Track Your Carbon Footprint')
    st.write('by : Muhammad Julizar')



    with st.form(key='form_parameters'):
        make = st.selectbox('Brand', ('ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW',
                                      'BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'DODGE', 'FIAT',
                                      'FORD', 'GMC', 'HONDA', 'HYUNDAI', 'INFINITI', 'JAGUAR', 'JEEP',
                                      'KIA', 'LAMBORGHINI', 'LAND ROVER', 'LEXUS', 'LINCOLN', 'MASERATI',
                                      'MAZDA', 'MERCEDES-BENZ', 'MINI', 'MITSUBISHI', 'NISSAN',
                                      'PORSCHE', 'RAM', 'ROLLS-ROYCE', 'SCION', 'SMART', 'SRT', 'SUBARU',
                                      'TOYOTA', 'VOLKSWAGEN', 'VOLVO', 'GENESIS', 'BUGATTI'), index=1)
        Vehicle_Class = st.selectbox('Vehicle Class', ('COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
                                                      'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL',
                                                      'SUV - STANDARD', 'VAN - CARGO', 'VAN - PASSENGER',
                                                      'PICKUP TRUCK - STANDARD', 'MINIVAN', 'SPECIAL PURPOSE VEHICLE',
                                                      'STATION WAGON - MID-SIZE', 'PICKUP TRUCK - SMALL'), index=1)
        transmission = st.radio('Transmission', ['Automatic', 'Manual'])
        Fuel_Type = st.selectbox('Fuel Type', ('Premium Gasoline', 'Regular Gasoline', 'Ethanol', 'Diesel'), index=1)
        Engine_Size = st.slider('Engine Size', min_value=1.0, max_value=8.4,value=1.0,step=0.1)
        Cylinders = st.number_input('Cylinders', min_value=1, max_value=16, value=1, step=1)
        Fuel_Consumption_Liter = st.number_input('Fuel Consumption (L/100 km)', min_value=1.0, max_value=26.0, value=1.0, step=0.5)
        Fuel_Consumption_mpg = st.number_input('Fuel Consumption (mpg)', min_value=11, max_value=70, value=11, step=1)
        st.markdown('---')

        submitted = st.form_submit_button('Predict')


    data_inf={
            'Make'           : make,
            'Vehicle Class'   : Vehicle_Class,
            'Transmission'    : transmission,
            'Fuel Type'       : Fuel_Type,
            'Engine Size(L)'     : Engine_Size,
            'Cylinders'       : Cylinders,
            'Fuel Consumption Comb (L/100 km)'  :Fuel_Consumption_Liter,
            'Fuel Consumption Comb (mpg)'    : Fuel_Consumption_mpg
            }


    data_inf = pd.DataFrame([data_inf])

    if submitted:       
        y_pred = pipeline_model_randomforest.predict(data_inf)

        st.write('The result is :')
        st.write(f'<p style="font-size:40px;">CO2 Emissions : {str(int(y_pred))} g/km</p>', unsafe_allow_html=True)
        

if __name__ == '__main__':
    run()