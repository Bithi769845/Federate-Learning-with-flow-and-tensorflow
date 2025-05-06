# import numpy as np

# def save_combined_data(client_data, filename='combined_data.npz'):
#     # Combine data for all clients
#     X_train_combined = np.concatenate([client_data[i][0] for i in range(len(client_data))], axis=0)
#     y_train_combined = np.concatenate([client_data[i][1] for i in range(len(client_data))], axis=0)
#     X_val_combined = np.concatenate([client_data[i][2] for i in range(len(client_data))], axis=0)
#     y_val_combined = np.concatenate([client_data[i][3] for i in range(len(client_data))], axis=0)

#     # Save combined data to a .npz file
#     np.savez_compressed(filename, X_train=X_train_combined, y_train=y_train_combined, X_val=X_val_combined, y_val=y_val_combined)
#     print(f"Combined data saved to {filename}")

    
# def load_combined_data(filename='combined_data.npz'):
#     # Load the combined data from the .npz file
#     data = np.load(filename)

#     # Unpack the data into variables
#     X_train_combined = data['X_train']
#     y_train_combined = data['y_train']
#     X_val_combined = data['X_val']
#     y_val_combined = data['y_val']

#     return X_train_combined, y_train_combined, X_val_combined, y_val_combined
