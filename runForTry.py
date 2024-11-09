import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# Definir SEBlock para que sea reconocido al cargar el modelo


class SEBlock(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(
            num_channels // self.reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(num_channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, num_channels))
        self.multiply = layers.Multiply()

    def call(self, inputs):
        squeeze = self.global_average_pooling(inputs)
        excite = self.dense1(squeeze)
        excite = self.dense2(excite)
        excite = self.reshape(excite)
        output = self.multiply([inputs, excite])
        return output

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


# Cargar el modelo incluyendo SEBlock como objeto personalizado
model = load_model("best_model.h5", custom_objects={"SEBlock": SEBlock})

# Definir las 12 columnas que el modelo espera como entrada
expected_columns = [
    "Store", "CompetitionDistance", "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
    "Promo2SinceYear", "StoreType_b", "StoreType_c",
    "StoreType_d", "Assortment_b", "Assortment_c"
]

# Definir las columnas que necesitan escalado
scaled_columns = ["CompetitionDistance", "CompetitionOpenSinceMonth",
                  "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"]

# Configurar un escalador
scaler = StandardScaler()

# Pedir al usuario los valores necesarios con el formato esperado


def solicitar_datos():
    sample_input = {
        'Store': int(input("Ingrese el número de tienda (Entero) (Store): ")),
        'CompetitionDistance': float(input("Ingrese la distancia a la competencia (Número decimal, en metros) (CompetitionDistance): ")),
        'CompetitionOpenSinceMonth': float(input("Mes en que abrió la competencia (Número decimal, 1 a 12) (CompetitionOpenSinceMonth): ")),
        'CompetitionOpenSinceYear': int(input("Año en que abrió la competencia (Entero, ej. 2010) (CompetitionOpenSinceYear): ")),
        'Promo2': int(input("¿La tienda tiene una promoción continua? (1 para sí, 0 para no) (Promo2): ")),
        'Promo2SinceWeek': float(input("Semana en que comenzó Promo2 (Número decimal, 1 a 52) (Promo2SinceWeek): ")),
        'Promo2SinceYear': float(input("Año en que comenzó Promo2 (Número decimal, ej. 2010) (Promo2SinceYear): ")),
        'StoreType_b': int(input("¿Es tienda de tipo B? (1 para sí, 0 para no) (StoreType_b): ")),
        'StoreType_c': int(input("¿Es tienda de tipo C? (1 para sí, 0 para no) (StoreType_c): ")),
        'StoreType_d': int(input("¿Es tienda de tipo D? (1 para sí, 0 para no) (StoreType_d): ")),
        'Assortment_b': int(input("¿Tiene surtido de tipo B? (1 para sí, 0 para no) (Assortment_b): ")),
        'Assortment_c': int(input("¿Tiene surtido de tipo C? (1 para sí, 0 para no) (Assortment_c): "))
    }
    return pd.DataFrame([sample_input])

# Función de preprocesamiento y predicción


def predecir_ventas(model, sample_input_df):
    # Escalar las columnas necesarias
    sample_input_scaled_part = scaler.fit_transform(
        sample_input_df[scaled_columns])

    # Crear DataFrame con las columnas escaladas
    sample_input_scaled_df = pd.DataFrame(
        sample_input_scaled_part, columns=scaled_columns)

    # Combinar con las columnas que no requieren escalado
    sample_input_final = sample_input_df.drop(
        columns=scaled_columns).join(sample_input_scaled_df)

    # Filtrar solo las columnas esperadas en el modelo
    sample_input_final = sample_input_final[expected_columns]

    # Redimensionar la entrada para la compatibilidad con CNN
    sample_input_cnn = np.expand_dims(sample_input_final.values, axis=2)

    # Realizar la predicción
    prediction = model.predict(sample_input_cnn)
    return prediction[0][0]


# Obtener datos del usuario y hacer la predicción
sample_input_df = solicitar_datos()
ventas_predichas = predecir_ventas(model, sample_input_df)
print(f"\nPredicción de ventas para la tienda: {ventas_predichas:.2f}")
