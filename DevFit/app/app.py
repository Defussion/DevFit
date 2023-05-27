from flask import Flask, render_template, request
import pandas as pd
from sklearn import tree


data = pd.read_csv('datos_entrenamiento.csv')


X = data.drop(['rutina', 'ejercicio1', 'ejercicio2', 'ejercicio3', 'ejercicio4', 'ejercicio5', 'ejercicio6'], axis=1)
y = data['rutina']


model = tree.DecisionTreeClassifier()
model.fit(X, y)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Obtener los datos del formulario enviado por el usuario
        altura = float(request.form['altura'])
        peso = float(request.form['peso'])
        edad = int(request.form['edad'])
        dias_entrenamiento = int(request.form['dias_entrenamiento'])
        proposito = request.form['proposito']
        genero = request.form['genero']
        nivel = request.form['nivel']

        # Realizar la predicción con el modelo de árbol de decisión
        datos_usuario = [[altura, peso, edad, dias_entrenamiento, proposito, genero, nivel]]
        rutina_predicha = model.predict(datos_usuario)[0]

        # Obtener los ejercicios correspondientes a la rutina predicha
        ejercicios = data.loc[data['rutina'] == rutina_predicha, 'ejercicio1':'ejercicio6'].values[0]


        return render_template('result.html', rutina=rutina_predicha, ejercicios=ejercicios)

    return render_template('index.html')

# Ejecutar el aplicativo web
if __name__ == '__main__':
    app.run(debug=True)