import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from FoldLayer import fold, soft_fold
import torch

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

fld = fold(2)
sft_fld = soft_fold(2)

points = np.random.rand(20, 2)

x1 = st.slider(label='x1', min_value=-1.0, max_value=1.0, 
          step=0.01, value=0.05, key="n_x1")

y1 = st.slider(label='y1', min_value=-1.0, max_value=1.0, 
          step=0.01, value=0.05, key="n_y1")

if (x1 == 0) and (y1 == 0) :
    st.warning("x and y cannot both be 0")
    x1 = 0.05
    y1 = 0.05

fig, ax = plt.subplots(figsize=(5,5))
ax.arrow(0, 0, x1, y1)
linspace = np.linspace(-1, 1, 200)
ax.plot(linspace, -(x1/y1)*(linspace - x1) + y1, color='red')
ax.scatter(points[:, 0], points[:, 1], color='blue')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
# ax.grid()
st.pyplot(fig, use_container_width=False)
