import dill as pickle
with open('model_v1.pk','rb') as f:
	loaded_model=pickle.load(f)
