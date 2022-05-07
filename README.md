# Conv_Classifier_For_Veggetables

This project is just a little image conv net for image classification of veggetables.
You must first download M Israk Ahmed's Vegetable Image Dataset on kaggle and locate this git code in the same folder as the "train" and "test" folders from the dataset.

- If you want to train and save the model, run main.py (note that the first time will be longer due to an initial step of data conversion and storage).
- If you want to test it on the whole test dataset, run test_model.py (also the first time will be longer for the same reasons).
- If you want to test the trained model on a single image, use the "guess" function in Veggetable_guesser.py by giving it the image path for ex :       guess('./test/Cabbage/1104.jpg')

An explanatory Jupyter Notebook is given to explain the coding steps.

