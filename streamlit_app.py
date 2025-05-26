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
Published in 1999, the fold-and-cut theorem states 
> Given any collection of straight edges,
> there exisits a flat folding and a line in that folding such
> that cutting along it results in the desired pattern of cuts. 
            
In other words, you can cut out any collection of polygons from a piece of paper by some number of folds and one straight cut. 
"This includes multiple disjoint, nested, and/or adjoining polygons" that need not be convex. The original paper was titled 
[Folding and One Straight Cut Suffice](https://www.imsc.res.in/~hbar/PDFs_2017/Paper_by_Demaine_Demaine_Lubiw.pdf), or in today's 
machine learning language, folding and cutting "are all you need." 
            
#### Connection to Machine Learning
One common task in machine learning is creating decision boundaries. What if these decision boundaries could be created by folding and cutting? 

### The Fold Layer
Fold layers generalize simple folds to higher dimensions, learning a hyperplane over which to "fold" the data. 
The regular fold layer is parameterized by $\mathbf{n}$, which is both the normal vector for the hyperplane and 
also a point through which the hyperplane passes. The fold layer is given by:  
""")

st.latex(r'''
\text{Fold}(\mathbf{x}) = \mathbf{x} - \mathbf{1}_{\{\mathbf{x} \cdot \mathbf{n} > \mathbf{n} \cdot \mathbf{n}\}} \eta  
 \left(1 - \frac{\mathbf{x} \cdot \mathbf{n}}{\mathbf{n} \cdot \mathbf{n}} \right) \mathbf{n}.''')

st.markdown("""
The stretch variable $\eta$ is a scalar set to 2 for a normal fold, though it can also be made a learnable parameter. 
An $\eta$ value greater than 2 would indicate stretching the data farther after folding. The $\mathbf{1}$ indicator function determines 
whether the data is on the "exterior" of the hyperplane (the side of the hyperplane that doesn't include the origin), 
in which case it gets folded, or if it's on the "interior" (the side of the hyperplane that contains the origin), 
in which case it stays put. We can also switch the indicator function to $\mathbf{1}_{\{\mathbf{x} \cdot \mathbf{n} < \mathbf{n} \cdot \mathbf{n}\}}$
in which case the layer "folds out" instead of "folding in."
""")

def create_fold_indicator_plot(fold_in) :
    fig, ax = plt.subplots(figsize=(8,1.5))
    linspace = np.linspace(-1, 1, 100)
    if fold_in : 
        ax.plot(linspace, linspace > 0, color='blue')
    else : 
        ax.plot(linspace, linspace < 0, color='blue')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    ax.set_xlabel("$\mathbf{x} \cdot \mathbf{n} - \mathbf{n} \cdot \mathbf{n}$")
    ax.set_title("Indicator function")
    return fig

# add a toggle for fold in / fold out 
if 'fold_in' not in st.session_state:
    st.session_state.fold_in = True

fold_in = st.toggle("Fold in", key='fold_in')

fig = create_fold_indicator_plot(fold_in)
st.pyplot(fig, use_container_width=False)

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
    st.session_state.x = 0.05
    st.session_state.y = 0.05
    st.rerun()

def create_plot(points, x1, y1):
    """Helper function to create consistent plots"""
    fig, ax = plt.subplots(figsize=(5,5))
    ax.arrow(0, 0, x1, y1, head_width=min(0.05, np.linalg.norm(np.array([x1,y1]))), 
                                          head_length=min(0.07, np.linalg.norm(np.array([x1,y1]))), 
                                          length_includes_head=True, color='dimgrey')
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
            fold.fold_in = fold_in
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

st.markdown("""
### The SoftFold Layer
""")


def create_softfold_indicator_plot(crease) :
    fig, ax = plt.subplots(figsize=(8,1.5))
    linspace = np.linspace(-3, 3, 100)
    ax.plot(linspace, 1 / (1 + np.exp(-linspace * crease)), color='blue')
    ax.set_xlabel("$\mathbf{x} \cdot \mathbf{n} - \mathbf{n} \cdot \mathbf{n}$")
    ax.set_title("How much $\mathbf{x}$ gets folded")
    return fig

# add a toggle for fold in / fold out 
if 'crease' not in st.session_state:
    st.session_state.crease = 1.0

crease = st.slider("Crease", -10.0, 10.0, 1.0, 0.01, key='crease')

fig = create_softfold_indicator_plot(crease)
st.pyplot(fig, use_container_width=False)

# Initialize session state for persistent points storage
if 'soft_points' not in st.session_state:
    points = torch.randn(35, 2) # restrict to be between -1 and 1
    st.session_state.soft_points = points[torch.prod((points < 1) & (points > -1),axis=1) == 1, :]
if 'soft_folded' not in st.session_state:
    st.session_state.soft_folded = False

# Create sliders with validation
col1, col2 = st.columns(2)
with col1:
    x2 = st.slider('x2', -1.0, 1.0, 0.05, 0.01)
with col2:
    y2 = st.slider('y2', -1.0, 1.0, 0.05, 0.01)

# Prevent both sliders from being zero
if x2 == 0 and y2 == 0:
    st.warning("x and y cannot both be 0, using 0.05 for both")
    st.session_state.x = 0.05
    st.session_state.y = 0.05
    st.rerun()

# Display the main plot
current_points = st.session_state.soft_points.detach().numpy() if st.session_state.folded else st.session_state.soft_points
fig = create_plot(current_points, x2, y2)
st.pyplot(fig, use_container_width=False)


col1, col2, col3 = st.columns(3)

with col1 : 
    if st.button("Soft Fold"):
        with st.spinner("Folding..."):
            softfold = SoftFold(2,crease=crease)
            softfold.n = torch.nn.Parameter(torch.tensor([x2, y2]))
            st.session_state.soft_points = softfold(st.session_state.soft_points)
            st.session_state.folded = True
        st.rerun()  # Force immediate update

with col2 : 
    if st.button("Reset and shuffle points", key='reset_soft'): 
        with st.spinner("Reseting and shuffling points..."):
            points = torch.randn(35, 2)
            st.session_state.soft_points = points[torch.prod((points < 1) & (points > -1),axis=1) == 1, :]
            st.session_state.folded = False
        st.rerun()  # Force immediate update