import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from FoldLayer import Fold, SoftFold
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

# Initialize session state for persistent points storage
if 'points' not in st.session_state:
    points = torch.randn(35, 2) # restrict to be between -1 and 1
    st.session_state.points = points[torch.prod((points < 1) & (points > -1),axis=1) == 1, :]
if 'folded' not in st.session_state:
    st.session_state.folded = False

# Create sliders with validation
col1, col2 = st.columns(2)
with col1:
    x1 = st.slider('x', -1.0, 1.0, 0.05, 0.01)
with col2:
    y1 = st.slider('y', -1.0, 1.0, 0.05, 0.01)

# Prevent both sliders from being zero
if x1 == 0 and y1 == 0:
    st.warning("x and y cannot both be 0, using 0.05 for both")
    st.session_state.n_x1 = 0.05
    st.session_state.n_y1 = 0.05
    st.rerun()

def create_plot(points, x1, y1):
    """Helper function to create consistent plots"""
    fig, ax = plt.subplots(figsize=(5,5))
    ax.arrow(0, 0, x1, y1, head_width=min(0.05, np.linalg.norm(np.array([x1,y1])), 
                                          head_length=min(0.07, np.linalg.norm(np.array([x1,y1]))), 
                                          length_includes_head=True, color='dimgrey'))
    linspace = np.linspace(-1, 1, 1000)
    
    if y1 == 0:
        ax.plot(x1 * np.ones_like(linspace), linspace, color='red')
    else:
        ax.plot(linspace, -(x1/y1)*(linspace - x1) + y1, color='red')
    
    ax.scatter(points[:, 0], points[:, 1], color='blue')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return fig

# Display the main plot
current_points = st.session_state.points.detach().numpy() if st.session_state.folded else st.session_state.points
fig = create_plot(current_points, x1, y1)
st.pyplot(fig, use_container_width=False)


col1, col2, col3 = st.columns(3)

with col1 : 
    if st.button("Fold"):
        with st.spinner("Folding..."):
            fold = Fold(2)
            fold.n = torch.nn.Parameter(torch.tensor([x1, y1]))
            st.session_state.points = fold(st.session_state.points)
            st.session_state.folded = True
        st.rerun()  # Force immediate update

with col2 : 
    if st.button("Reset and shuffle points") :
        with st.spinner("Reseting and shuffling points..."):
            points = torch.randn(35, 2)
            st.session_state.points = points[torch.prod((points < 1) & (points > -1),axis=1) == 1, :]
            st.session_state.folded = False
        st.rerun()  # Force immediate update
