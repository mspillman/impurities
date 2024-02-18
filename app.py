import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F


import plotly.express as px
import pandas as pd
from model import ConvNeXtTransformer



st.set_page_config(page_title='PXRD impurity detection', page_icon = ":test_tube:",
                    layout = 'centered', initial_sidebar_state = 'auto')

st.title("PXRD impurity detection")

device = torch.device("cpu")
dtype = torch.float32

model = ConvNeXtTransformer(depths=[2,2,6,2],
                            dims=[4,8,16,32],
                            transformer_layers=8,
                            attention_heads=4,
                            drop_path_prob=0,
                            dropout=0,
                        ).to(device)


state_dict = torch.load("Impurity_detection_ConvNeXt_Transformer.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict["model_state_dict"])
model.eval()

tt_to_d = lambda tt, lam: lam/(2*np.sin(tt/2))
d_to_tt = lambda d, lam: 2*np.arcsin(lam/(2*d))

with st.sidebar:
    option = st.radio("Options",["App","Instructions","About"])
    with open("colours.txt") as f:
        colours = f.readlines()
    colours = "".join([x.strip() for x in colours]).split(",")
    plot_xrd_colour = st.selectbox("Select colour for PXRD data", options=colours, index=135).strip()
    plot_model_colour = st.selectbox("Select colour for model prediction", options=colours, index=138).strip()

if "Instructions" in option:
    with open("instructions.md") as f:
        instructions = f.readlines()
    st.markdown("\n".join(instructions))
elif option == "About":
    st.write("This web app was written by [Mark Spillman](https://mspillman.github.io/blog/).")
    st.write("You can find the code [here](https://github.com/mspillman/impurities)")
    st.write("And a blog post describing the work [here](https://mspillman.github.io/blog/posts/2024-02-17-Detecting-impurities-in-PXRD-data.html)")
else:
    st.write("### Upload data in xye format")
    st.write("You can upload multiple files simultaneously")

    uploaded_file = st.file_uploader("Choose a file",["xye","xy"],
                                        accept_multiple_files=True)
    col1, col2 = st.columns(2)
    with col1:
        enter_wavelength = st.checkbox("Enter wavelength manually",False)
    with col2:
        if enter_wavelength:
            wavelength = st.number_input("Wavelength", value=1.54056, format="%.6f", min_value=0.0)
        else:
            wavelength = None
    datadim = 2048
    patchsize = 16
    if len(uploaded_file) > 0:
        # To read file as bytes:
        xye = []
        names = []
        for item in uploaded_file:
            data = item.getvalue().decode().split("\n")
            pos, i, sig = [], [], []
            lam = None
            for line in data:
                line = line.split()
                if len(line) == 1:
                    if not enter_wavelength:
                        lam = float(line[0])
                if len(line) > 1:
                    pos.append(float(line[0]))
                    i.append(float(line[1]))
                    #sig.append(float(line[2]))
            if enter_wavelength:
                lam = wavelength
            xye.append([pos, i, lam])
            names.append(item.name)
        for i, x in enumerate(xye):
            pos, intensity = x[0], x[1]
            lam = x[2]
            pos = np.array(pos)
            intensity = np.array(intensity)
            intensity -= intensity.min()
            intensity /= intensity.max()
            data = np.linspace(4,44,datadim)
            if lam is None:
                lam = 1.54056
                if not enter_wavelength:
                    st.write("Assuming Cu Ka1 data")
            d_obs = tt_to_d(np.radians(pos), lam)
            d_desired = tt_to_d(np.radians(np.copy(data)), 1.54046)
            d_obs = d_obs[::-1]
            d_desired = d_desired[::-1]
            intensity = intensity[::-1]
            newint = np.interp(d_desired, d_obs, intensity)
            d_obs = d_obs[::-1]
            d_desired = d_desired[::-1]
            intensity = intensity[::-1]
            newint = newint[::-1]

            data = torch.from_numpy(data).type(dtype)
            newint = torch.tensor(newint.tolist()).type(dtype).unsqueeze(0).unsqueeze(0)

            predicted, peak_prediction = model(newint.reshape(1, 1, datadim).to(device))

            df = pd.DataFrame()
            df["2theta"] = np.degrees(d_to_tt(d_desired, lam))
            df["Intensities"] = newint.squeeze().numpy()
            df["prob_impure"] = F.sigmoid(peak_prediction.detach().squeeze().reshape(datadim))


            fig = px.line(df, x="2theta", y=["Intensities","prob_impure"],color_discrete_sequence=[plot_xrd_colour,plot_model_colour])
            fig.update_layout(yaxis_title='Intensity')
            st.markdown("##### "+names[i])
            st.write(f"Using wavelength of {lam} Å")
            st.write(f"Probability of impure data = {100*F.sigmoid(predicted).squeeze().item():.2f} %")
            st.plotly_chart(fig)