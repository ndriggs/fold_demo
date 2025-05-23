import streamlit as st
import matplotlib.pyplot as plt

# Potential titles 
# MLP Enhancing Fold Layers
# Parameter Efficient Fold Layers
# Origami Inspired Neural Networks
# A New Mathematically-Inspired Nonlinear Layer
# Fold-and-cut Theorem Inspired Neural Networks
st.title("Fold-and-cut Theorem Inspired Neural Networks")

st.markdown("""
[Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (KANs) are an alternative neural network architecture inspired by the 
Kolmogorov-Arnold representation theorem. Here we present another novel architecture design inspired by a theorem in geometry known
as the fold-and-cut theorem. 
            
### The Fold-and-cut Theorem
The fold-and-cut theorem states 
""")

def plot_point() :
    fig, ax = plt.subplots()
    ax.arrow(0,0,st.session_state.n_x, st.session_state.n_y)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid()
    st.pyplot(fig)

st.slider(label='x', min_value=-1.0, max_value=1.0, 
          step=0.01, value=0.0, key="n_x", on_change=plot_point)

st.slider(label='y', min_value=-1.0, max_value=1.0, 
          step=0.01, value=0.0, key="n_y", on_change=plot_point)