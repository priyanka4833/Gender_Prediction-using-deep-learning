# Gender_Prediction-using-deep-learning
#This model predicts the gender based on first name and last name developed using LSTM.


#before running the model,check requirement.txt for packages and version
#command to train the model

python3 gender_prediction_AI_model.py traindata_path,model_path

e.g: python3 gender_prediction_AI_model.py "/home/priyankagore/Downloads/Machine-Learning-Skills-Test/Machine Learning Skills Test/name_gender.csv" "/home/priyankagore/Downloads/Machine-Learning-Skills-Test/Machine Learning Skills Test/ML_Model/"

#trained model will be saved in model_path
#the model is giving 90% accuracy.
#####################################################################################################################################################
#before running the model,check requirement.txt for packages and version
#command to serve or predict  the model

python3 predict_gender.py <name_to_predict>,model_path

e.g: python3 predict_gender.py "Priyanka" "/home/priyankagore/Downloads/Machine-Learning-Skills-Test/Machine Learning Skills Test/model/lstm_gender_model.sav"



#for flask application,refer flask_application folder

