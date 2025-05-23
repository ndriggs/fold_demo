import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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

x = st.slider(label='x', min_value=-1.0, max_value=1.0, 
          step=0.01, value=0.0, key="n_x")

y = st.slider(label='y', min_value=-1.0, max_value=1.0, 
          step=0.01, value=0.0, key="n_y")

fig, ax = plt.subplots()
ax.arrow(0, 0, x, y)
# plot a line perpendicular to the arrow that goes through the arrow head
linspace = np.linspace(-1, 1, 200)
ax.plot(linspace, -(x/y)*(linspace - x) + y, color='red')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
# ax.grid()
st.pyplot(fig)