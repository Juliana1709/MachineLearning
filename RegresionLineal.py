import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generar el mismo conjunto de datos sintéticos (el mismo proceso que antes)
num_samples = 1000
hora_del_dia = np.random.randint(0, 24, num_samples)
dia_de_la_semana = np.random.randint(0, 7, num_samples)
estaciones = [f'Estacion_{i}' for i in range(1, 11)]
estacion_origen = np.random.choice(estaciones, num_samples)
estacion_destino = np.random.choice(estaciones, num_samples)
duracion_viaje = np.random.randint(10, 61, num_samples)
numero_pasajeros = np.random.randint(1, 101, num_samples)
tipos_vehiculo = ['autobus', 'tren', 'metro']
tipo_vehiculo = np.random.choice(tipos_vehiculo, num_samples)
condiciones_meteorologicas = ['soleado', 'lluvioso', 'nublado']
condicion_meteorologica = np.random.choice(condiciones_meteorologicas, num_samples)
eventos_especiales = np.random.choice([0, 1], num_samples)

dataset = pd.DataFrame({
    'Hora_del_dia': hora_del_dia,
    'Dia_de_la_semana': dia_de_la_semana,
    'Estacion_origen': estacion_origen,
    'Estacion_destino': estacion_destino,
    'Duracion_viaje': duracion_viaje,
    'Numero_pasajeros': numero_pasajeros,
    'Tipo_vehiculo': tipo_vehiculo,
    'Condicion_meteorologica': condicion_meteorologica,
    'Evento_especial': eventos_especiales
})

# Convertir las variables categóricas a variables dummy (one-hot encoding)
dataset_encoded = pd.get_dummies(dataset, columns=['Estacion_origen', 'Estacion_destino', 'Tipo_vehiculo', 'Condicion_meteorologica'])

# Aplicar KMeans
kmeans = KMeans(n_clusters=3)  
kmeans.fit(dataset_encoded)  

# Añadir etiquetas de clusters al conjunto de datos
dataset['Cluster'] = kmeans.labels_

# Visualizar los resultados
plt.scatter(dataset['Hora_del_dia'], dataset['Duracion_viaje'], c=dataset['Cluster'], cmap='viridis')
plt.xlabel('Hora del día')
plt.ylabel('Duración del viaje')
plt.title('Agrupación KMeans')
plt.show()
