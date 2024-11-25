import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Ruta a los datos (ajusta estas rutas a tu estructura de archivos)
train_data_path = './data/train.csv'  # Datos de entrenamiento
# Información adicional de las tiendas
store_data_path = './data/store_cleaned.csv'

# Cargar los datos
train = pd.read_csv(train_data_path)
store = pd.read_csv(store_data_path)

# Preprocesar los datos del dataset `store`
store['CompetitionDistance'].fillna(
    store['CompetitionDistance'].median(), inplace=True)
store['CompetitionOpenSinceMonth'].fillna(1, inplace=True)
store['CompetitionOpenSinceYear'].fillna(
    store['CompetitionOpenSinceYear'].median(), inplace=True)
store['Promo2SinceWeek'].fillna(1, inplace=True)
store['Promo2SinceYear'].fillna(
    store['Promo2SinceYear'].median(), inplace=True)

categorical_columns = ['StoreType', 'Assortment', 'PromoInterval']
categorical_columns = [
    col for col in categorical_columns if col in store.columns]

if categorical_columns:
    store = pd.get_dummies(store, columns=categorical_columns, drop_first=True)

# Merge de store con train
train['Date'] = pd.to_datetime(train['Date'], dayfirst=True)
train['Month'] = train['Date'].dt.month
train['Year'] = train['Date'].dt.year
train = train.merge(store, on='Store', how='left')

# Eliminar filas donde Sales es 0
train = train[train['Sales'] > 0]

# Crear nuevas características
train['Promo_Customers'] = train['Promo'] * train['Customers']
train['Open_Customers'] = train['Open'] * train['Customers']

# Seleccionar columnas finales
columns_to_keep = [
    'DayOfWeek', 'Customers', 'Open', 'Promo', 'SchoolHoliday', 'Month',
    'Promo_Customers', 'Open_Customers', 'Sales'
]

processed_categorical_columns = [
    col for col in train.columns if col.startswith(('StoreType', 'Assortment', 'PromoInterval'))
]
columns_to_keep.extend(processed_categorical_columns)

state_holiday_columns = [
    col for col in train.columns if col.startswith('StateHoliday')]
columns_to_keep.extend(state_holiday_columns)

columns_to_keep = [col for col in columns_to_keep if col in train.columns]
train = train[columns_to_keep]

# Separar características (X) y variable objetivo (y)
X_train = train.drop(['Sales'], axis=1)
y_train = train['Sales']

# Definir las columnas que el modelo espera como entrada
expected_columns = [
    'DayOfWeek', 'Customers', 'Open', 'Promo', 'SchoolHoliday', 'Month',
    'Promo_Customers', 'Open_Customers', 'StoreType_b', 'StoreType_c',
    'StoreType_d', 'Assortment_b', 'Assortment_c',
    'PromoInterval_Jan,Apr,Jul,Oct', 'PromoInterval_Mar,Jun,Sept,Dec',
    'PromoInterval_NoPromo', 'StateHoliday'
]

# Definir las columnas que necesitan escalado
scaled_columns = ['Customers', 'Promo_Customers', 'Open_Customers']

# Configurar un escalador
scaler = MinMaxScaler()

# Ajustar el escalador con los datos de entrenamiento
scaler.fit(X_train[scaled_columns])

# Pedir al usuario los valores necesarios con el formato esperado
def solicitar_datos():
    sample_input = {
        'DayOfWeek': int(input("Día de la semana (1 a 7) (DayOfWeek): ")),
        'Customers': float(input("Número de clientes esperados (decimal) (Customers): ")),
        'Open': int(input("¿La tienda está abierta? (1 para sí, 0 para no) (Open): ")),
        'Promo': int(input("¿Hay una promoción activa? (1 para sí, 0 para no) (Promo): ")),
        'SchoolHoliday': int(input("¿Es un día festivo escolar? (1 para sí, 0 para no) (SchoolHoliday): ")),
        'Month': int(input("Mes del año (1 a 12) (Month): ")),
        'Promo_Customers': float(input("Clientes afectados por la promoción (decimal) (Promo_Customers): ")),
        'Open_Customers': float(input("Clientes afectados por la apertura (decimal) (Open_Customers): ")),
        'StoreType_b': int(input("¿Es tienda de tipo B? (1 para sí, 0 para no) (StoreType_b): ")),
        'StoreType_c': int(input("¿Es tienda de tipo C? (1 para sí, 0 para no) (StoreType_c): ")),
        'StoreType_d': int(input("¿Es tienda de tipo D? (1 para sí, 0 para no) (StoreType_d): ")),
        'Assortment_b': int(input("¿Tiene surtido de tipo B? (1 para sí, 0 para no) (Assortment_b): ")),
        'Assortment_c': int(input("¿Tiene surtido de tipo C? (1 para sí, 0 para no) (Assortment_c): ")),
        'PromoInterval_Jan,Apr,Jul,Oct': int(input("¿Promoción en Enero, Abril, Julio, Octubre? (1 para sí, 0 para no): ")),
        'PromoInterval_Mar,Jun,Sept,Dec': int(input("¿Promoción en Marzo, Junio, Septiembre, Diciembre? (1 para sí, 0 para no): ")),
        'PromoInterval_NoPromo': int(input("¿No hay promoción? (1 para sí, 0 para no): ")),
        'StateHoliday': int(input("¿Es un día festivo estatal? (1 para sí, 0 para no) (StateHoliday): "))
    }
    return pd.DataFrame([sample_input])

# Función de preprocesamiento y predicción
def predecir_ventas(model, scaler, sample_input_df):
    """
    Preprocesa los datos ingresados y realiza una predicción con el modelo cargado.

    Args:
        model (keras.Model): Modelo cargado para realizar predicciones.
        scaler (MinMaxScaler): Escalador ajustado a las características que requieren normalización.
        sample_input_df (DataFrame): Datos de entrada proporcionados por el usuario.

    Returns:
        float: Predicción de ventas.
    """
    # Escalar las columnas necesarias
    sample_input_scaled_part = scaler.transform(
        sample_input_df[scaled_columns])

    # Crear DataFrame con las columnas escaladas
    sample_input_scaled_df = pd.DataFrame(
        sample_input_scaled_part, columns=scaled_columns)

    # Combinar con las columnas que no requieren escalado
    sample_input_final = sample_input_df.drop(
        columns=scaled_columns).join(sample_input_scaled_df)

    # Filtrar solo las columnas esperadas en el modelo
    sample_input_final = sample_input_final[expected_columns]

    # Redimensionar la entrada para la compatibilidad con LSTM
    sample_input_rnn = np.expand_dims(sample_input_final.values, axis=1)

    # Realizar la predicción
    prediction = model.predict(sample_input_rnn)
    return prediction[0][0]


# Cargar el modelo
model_path = './final_rnn_model_v2.h5'
model = load_model(model_path)

# Obtener datos del usuario y hacer la predicción
sample_input_df = solicitar_datos()
ventas_predichas = predecir_ventas(model, scaler, sample_input_df)
print(f"\nPredicción de ventas para la tienda: {ventas_predichas:.2f}")
