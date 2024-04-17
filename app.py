import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

def cyclic_loss(lr, y_pred):
    pass

with tf.keras.utils.custom_object_scope({'cyclic_loss': cyclic_loss}):
    # Load the model with the custom loss function
    model = tf.keras.models.load_model("HSR_Net_US2_1000e_evaluated_best.h5")

def single_image_run(HSI, MSI, model):

    IMAGE_SIZE_MSI = 320
    IMAGE_SIZE_HSI = 64
    NUM_BANDS = 40

    HSI_bands = []
    HSI_arrays = []
    MSI_bands = []
    MSI_arrays = []

    for i in range(0, NUM_BANDS):
        HSI_bands.append(HSI[i])
    HSI_array = np.dstack(HSI_bands)
    HSI_arrays.append(HSI_array)

    for i in range(MSI.shape[0]):
        MSI_bands.append(MSI[i])
    MSI_array = np.dstack(MSI_bands)
    MSI_arrays.append(MSI_array)

    MSI_arrays = np.array(MSI_arrays)
    HSI_arrays = np.array(HSI_arrays)

    pred = model.predict([
        HSI_arrays.reshape(1, IMAGE_SIZE_HSI, IMAGE_SIZE_HSI, NUM_BANDS),
        MSI_arrays.reshape(1, IMAGE_SIZE_MSI, IMAGE_SIZE_MSI, 3),
    ])
    pred = np.transpose(pred[0], (2, 0, 1))
    return pred

def main():
    st.title("Super Resolution using Image Fusion of HSI and MSI")

    st.write("Upload the HSI and MSI files to perform super-resolution using image fusion.")

    # Upload HSI and MSI files
    hsi_file = st.file_uploader("Upload HSI file", type=["npy"])
    msi_file = st.file_uploader("Upload MSI file", type=["npy"])

    if hsi_file is not None and msi_file is not None:
        # Load the HSI and MSI images
        hsi = np.array(hsi_file)
        msi = np.array(msi_file)

        # Perform super-resolution using image fusion
        predicted_hsi = single_image_run(hsi, msi, model)

        # Display the images in the same column
        col1, col2 = st.columns(2)
        with col1:
            st.write("Input HSI (First Band):")
            fig, ax = plt.subplots()
            sns.heatmap(hsi[0], ax=ax)
            st.pyplot(fig)
        with col2:
            st.write("Predicted HSI (First Band):")
            fig, ax = plt.subplots()
            sns.heatmap(predicted_hsi[0], ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()