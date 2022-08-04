import pickle
        
# set the variable 'file_empty' to True when trained_models.pkl is empty or when you what to delete the content. 
# To delete the content insert 'yes' when prompt ask if you want to store (related to the function 
# store_models).    
def load_store_models(file_empty=False, agent_copy=None):
    save_filename = "keras_models/trained_models.pkl"

    # if the file is not empty, we load its content into a variable.
    if file_empty == False:
        with open(save_filename, "rb") as f:
            trained_models = pickle.load(f)
        
    # Thanks to 'store' we can decide whether to store or not the just trained model.
    store = str("no")
    store = input("\nUpdate the description of the model!!\nDo you want to save this model to the external file? \nyes or no")

    if store == "yes":
        if file_empty:
            trained_models = []
        else:
            trained_models.append(agent_copy)

        with open(save_filename, 'wb') as f:
            pickle.dump(trained_models, f, pickle.HIGHEST_PROTOCOL)
            
    return trained_models