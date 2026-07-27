

import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Clasificación de Estrellas", page_icon="⭐", layout="centered")
st.title("Clasificación de Tipo de Estrella")
st.write(
    "Esta aplicación usa dos modelos entrenados (**kNN** y **MLP**) para predecir "
    "el tipo de una estrella a partir de sus propiedades físicas."
)


@st.cache_resource
def cargar_artefactos():
    modelo_knn = joblib.load("modelo_knn.joblib")
    modelo_mlp = joblib.load("modelo_mlp.joblib")
    type_labels = joblib.load("type_labels.joblib")
    metadata = joblib.load("metadata.joblib")
    return modelo_knn, modelo_mlp, type_labels, metadata

modelo_knn, modelo_mlp, type_labels, metadata = cargar_artefactos()

feature_num = metadata["feature_num"]
feature_cat = metadata["feature_cat"]
rangos = metadata["ranges"]
categorias = metadata["categories"]

st.sidebar.header("Desempeño de los modelos (test)")
st.sidebar.metric("Exactitud kNN", f"{metadata['acc_knn']*100:.1f}%")
st.sidebar.metric("Exactitud MLP", f"{metadata['acc_mlp']*100:.1f}%")
st.sidebar.caption(f"kNN entrenado con k = {metadata['best_k']}")


st.header("Ingresa los datos de la estrella")

col1, col2 = st.columns(2)

with col1:
    temperatura = st.number_input(
        "Temperatura (K)",
        min_value=float(rangos["Temperature (K)"][0]),
        max_value=float(rangos["Temperature (K)"][1]) * 1.2,
        value=5778.0,
        step=100.0,
    )
    luminosidad = st.number_input(
        "Luminosidad (L/Lo)",
        min_value=0.0,
        value=1.0,
        step=0.1,
        format="%.5f",
        help="Luminosidad relativa a la del Sol (Lo = 1)",
    )

with col2:
    radio = st.number_input(
        "Radio (R/Ro)",
        min_value=0.0,
        value=1.0,
        step=0.1,
        format="%.4f",
        help="Radio relativo al del Sol (Ro = 1)",
    )
    magnitud = st.number_input(
        "Magnitud absoluta (Mv)",
        min_value=float(rangos["Absolute magnitude(Mv)"][0]) * 1.2,
        max_value=float(rangos["Absolute magnitude(Mv)"][1]) * 1.2,
        value=4.83,
        step=0.1,
    )

col3, col4 = st.columns(2)
with col3:
    color = st.selectbox("Color de la estrella", options=categorias["Star color"])
with col4:
    clase_espectral = st.selectbox("Clase espectral", options=categorias["Spectral Class"])

modelo_elegido = st.radio("¿Qué modelo quieres usar?", options=["kNN", "MLP", "Ambos"], horizontal=True)


if st.button("🔮 Predecir tipo de estrella", type="primary"):


    entrada = pd.DataFrame([{
        "Temperature (K)": temperatura,
        "Luminosity(L/Lo)": luminosidad,
        "Radius(R/Ro)": radio,
        "Absolute magnitude(Mv)": magnitud,
        "Star color": color,
        "Spectral Class": clase_espectral,
    }])

    st.subheader("Resultado")

    st.markdown("**Datos ingresados (variables predictoras):**")
    resumen_entrada = pd.DataFrame({
        "Variable": ["Temperatura (K)", "Luminosidad (L/Lo)", "Radio (R/Ro)",
                     "Magnitud absoluta (Mv)", "Color", "Clase espectral"],
        "Valor": [temperatura, luminosidad, radio, magnitud, color, clase_espectral],
    })
    st.dataframe(resumen_entrada, hide_index=True, use_container_width=True)

    def mostrar_resultado(nombre_modelo, modelo):
        pred = modelo.predict(entrada)[0]
        proba = modelo.predict_proba(entrada)[0]
        # Clase predicha, destacada
        st.success(f"**{nombre_modelo}** predice →  **{type_labels[pred]}** "
                   f"(confianza: {proba[pred]*100:.1f}%)")
        tabla_proba = pd.DataFrame({
            "Tipo de estrella": [type_labels[i] for i in sorted(type_labels)],
            "Probabilidad": [f"{p*100:.1f}%" for p in proba],
        })
        st.dataframe(tabla_proba, hide_index=True, use_container_width=True)
        st.markdown(f"🏆 **Clase ganadora ({nombre_modelo}):** {type_labels[pred]}")

    if modelo_elegido in ("kNN", "Ambos"):
        mostrar_resultado("kNN", modelo_knn)
    if modelo_elegido in ("MLP", "Ambos"):
        mostrar_resultado("MLP", modelo_mlp)

st.divider()
st.caption(
    "Modelos entrenados sobre el dataset 'Star Type Classification (NASA)' "
    "con kNN y un Perceptrón Multicapa (MLP), usando scikit-learn."
)