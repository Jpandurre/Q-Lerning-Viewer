import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Q-Learning Visual Pro", layout="wide")

st.title("Q-Learning Visual Interactivo ICI")

# ----------------------------
# Estados y conexiones
# ----------------------------

states = ["A","B","C","D","E","F","G","H","I","J","K","L","M","Ñ",
          "N","O","P","Q","R","S","T","U","V"]

connections = [
    ("A","B"), ("A","C"), ("A","D"),
    ("B","C"),
    ("C","D"), ("C","K"),
    ("D","F"), ("D","E"),
    ("E","G"),
    ("G","H"), ("G","I"), ("G","J"),
    ("J","T"),
    ("K","L"), ("K","M"), ("K","Ñ"), ("K","N"),
    ("N","P"), ("N","O"), ("N","S"),
    ("O","R"),
    ("P","R"), ("P","Q"),
    ("R","U"),
    ("U","V"),
    ("T","U")
]

n = len(states)
idx = {state:i for i,state in enumerate(states)}

# ----------------------------
# Diagrama inicial
# ----------------------------

st.markdown("## Entorno")

G_preview = nx.Graph()
G_preview.add_edges_from(connections)
pos_preview = nx.spring_layout(G_preview, seed=42)

plt.figure(figsize=(7,5))
nx.draw(G_preview, pos_preview, with_labels=True, node_color="#4dabf7")
st.pyplot(plt)
plt.close()

st.markdown("---")

# ----------------------------
# Selección
# ----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    goal_state = st.selectbox("Estado Objetivo", states)

with col2:
    start_state = st.selectbox("Estado Inicial", states)

with col3:
    st.metric("Total Estados", n)

goal = idx[goal_state]
start = idx[start_state]

st.markdown("---")

# ----------------------------
# Crear R base (POSITIVO)
# ----------------------------

R = np.zeros((n,n))

for a,b in connections:
    i = idx[a]
    j = idx[b]
    R[i,j] = 1     # Movimiento normal positivo
    R[j,i] = 1

R[goal, goal] = 100

# ----------------------------
# MATRIZ EDITABLE (ANTES DEL ENTRENAMIENTO)
# ----------------------------

st.header("Matriz de Recompensas (Editable)")

df_R = pd.DataFrame(R, index=states, columns=states)

df_R_editable = st.data_editor(
    df_R,
    use_container_width=True,
    num_rows="fixed"
)

# Convertimos a numpy actualizado
R = df_R_editable.values

st.markdown("---")

# ----------------------------
# ENTRENAMIENTO
# ----------------------------

gamma = 0.75
alpha = 0.9
episodes = 4000

Q = np.zeros((n,n))

for _ in range(episodes):
    state = np.random.randint(0,n)

    # Solo acciones donde R != 0 (0 = no conexión)
    actions = np.where(R[state] != 0)[0]

    if len(actions) == 0:
        continue

    action = np.random.choice(actions)

    TD = R[state, action] + gamma*np.max(Q[action]) - Q[state, action]
    Q[state, action] += alpha*TD

# ----------------------------
# Botón
# ----------------------------

st.markdown("## Ejecutar Simulación")

start_sim = st.button("▶ Iniciar Simulación", use_container_width=True)

st.markdown("---")

# ----------------------------
# SIMULACIÓN
# ----------------------------

if start_sim:

    current = start
    path = [states[current]]
    total_reward = 0

    col_graph, col_table = st.columns([1.3, 1])

    graph_placeholder = col_graph.empty()
    table_placeholder = col_table.empty()
    explanation_placeholder = st.empty()

    G = nx.Graph()
    G.add_edges_from(connections)
    pos = nx.spring_layout(G, seed=42)

    steps_data = []

    for step in range(50):

        next_state = np.argmax(Q[current])

        if Q[current, next_state] == 0:
            break

        reward = R[current, next_state]
        max_q_next = np.max(Q[next_state])
        old_q = Q[current, next_state]

        TD = reward + gamma * max_q_next - old_q
        new_q = old_q + alpha * TD

        total_reward += reward

        steps_data.append({
            "Paso": step,
            "Estado Actual": states[current],
            "Siguiente Estado": states[next_state],
            "Recompensa": reward,
            "TD Error": round(TD,2),
            "Q nuevo": round(new_q,2),
            "Recompensa acumulada": total_reward
        })

        table_placeholder.dataframe(pd.DataFrame(steps_data), use_container_width=True)

        explanation_placeholder.markdown(f"""
### Paso {step}

TD = {reward} + ({gamma}) × ({round(max_q_next,2)}) − {round(old_q,2)}

Q_nuevo = {round(old_q,2)} + ({alpha}) × ({round(TD,2)})

Recompensa acumulada: **{total_reward}**
""")

        path.append(states[next_state])
        current = next_state

        plt.figure(figsize=(6,5))
        nx.draw(G, pos, with_labels=True, node_color='#4dabf7')

        edges_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges_path, edge_color='#ff4b4b', width=3)

        graph_placeholder.pyplot(plt)
        plt.close()

        time.sleep(0.8)

        if current == goal:
            break

    st.success("Objetivo alcanzado correctamente")
