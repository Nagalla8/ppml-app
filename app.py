import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import pickle
import os
from dotenv import load_dotenv
import hashlib
import base64

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Privacy-Preserving ML",
    page_icon="🔒",
    layout="wide"
)

# Title and description
st.title("🔒 Privacy-Preserving Machine Learning")
st.subheader("Privacy Protection Techniques")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "choose a section",
    [ "Data Privacy", "Differential Privacy", "Federated Learning", "Homomorphic Encryption"]
)


# Data Privacy page
if page == "Data Privacy":
    st.markdown("## Data Privacy Techniques")
    
    technique = st.selectbox(
        "Select a technique to explore",
        ["Data Anonymization", "Data Masking"]
    )
    
    if technique == "Data Anonymization":
        st.markdown("### Data Anonymization")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Name': ['John Smith', 'Jane Doe', 'Robert Johnson'],
            'Age': [34, 28, 45],
            'Email': ['john.s@example.com', 'jane.d@example.com', 'robert.j@example.com'],
            'SSN': ['123-45-6789', '987-65-4321', '456-78-9123'],
            'Medical': ['Diabetes', 'None', 'Hypertension'],
            'Salary': [72000, 65000, 81000]
        })
        
        st.write("Original Data:")
        st.dataframe(sample_data)
        
        if st.button("Anonymize Data"):
            # Perform anonymization
            anonymized_data = sample_data.copy()
            
            # Anonymize direct identifiers
            anonymized_data['Name'] = ['Person-' + hashlib.md5(name.encode()).hexdigest()[:6] 
                                      for name in anonymized_data['Name']]
            anonymized_data['Email'] = ['user' + str(i+1) + '@example.com' for i in range(len(anonymized_data))]
            anonymized_data['SSN'] = ['XXX-XX-' + ssn.split('-')[2] for ssn in anonymized_data['SSN']]
            
            # Generalize quasi-identifiers
            anonymized_data['Age'] = [age - (age % 5) for age in anonymized_data['Age']]  # Age buckets
            anonymized_data['Salary'] = [salary - (salary % 10000) for salary in anonymized_data['Salary']]  # Salary buckets
            
            st.write("Anonymized Data:")
            st.dataframe(anonymized_data)
            
            st.success("Sensitive information has been anonymized while preserving data utility")
    
    elif technique == "Data Masking":
        st.markdown("### Data Masking")
        
        # Sample text with sensitive info
        sample_text = """
        Patient John Smith (DOB: 05/12/1987, SSN: 123-45-6789) was admitted on 03/15/2023.
        Phone: (555) 123-4567, Email: john.smith@example.com
        Credit Card: 4111-2222-3333-4444, Exp: 05/25
        """
        
        st.text_area("Original Text", sample_text, height=100)
        
        mask_types = st.multiselect(
            "Select information to mask",
            ["Names", "Dates", "SSN", "Email", "Phone", "Credit Card"],
            default=["SSN", "Credit Card"]
        )
        
        if st.button("Apply Masking"):
            masked_text = sample_text
            
            if "Names" in mask_types:
                masked_text = masked_text.replace("John Smith", "[REDACTED]")
            
            if "Dates" in mask_types:
                masked_text = masked_text.replace("05/12/1987", "XX/XX/XXXX")
                masked_text = masked_text.replace("03/15/2023", "XX/XX/XXXX")
                masked_text = masked_text.replace("05/25", "XX/XX")
            
            if "SSN" in mask_types:
                masked_text = masked_text.replace("123-45-6789", "XXX-XX-XXXX")
            
            if "Email" in mask_types:
                masked_text = masked_text.replace("john.smith@example.com", "xxxxx@xxxxx.com")
            
            if "Phone" in mask_types:
                masked_text = masked_text.replace("(555) 123-4567", "(XXX) XXX-XXXX")
            
            if "Credit Card" in mask_types:
                masked_text = masked_text.replace("4111-2222-3333-4444", "XXXX-XXXX-XXXX-4444")
            
            st.text_area("Masked Text", masked_text, height=100)
            st.success("Masking applied successfully!")

# Differential Privacy page
elif page == "Differential Privacy":
    st.markdown("## Differential Privacy")
    
    st.markdown("""
    Differential Privacy adds calibrated noise to protect individual data points 
    while allowing accurate aggregate analysis.
    """)
    
    # Create sample dataset
    np.random.seed(42)
    income_data = np.random.lognormal(mean=11, sigma=0.7, size=1000)  # log-normal distribution for incomes
    
    # Display raw statistics
    st.subheader("Raw Statistics (No Privacy)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Income", f"${income_data.mean():.2f}")
    with col2:
        st.metric("Max Income", f"${income_data.max():.2f}")
    
    # Differential privacy parameters
    st.subheader("Apply Differential Privacy")
    
    epsilon = st.slider("Privacy Budget (ε)", 0.01, 2.0, 0.1, 0.01, 
                      help="Lower values = more privacy, less accuracy")
    
    # Function to add noise
    def add_laplace_noise(value, sensitivity, eps):
        scale = sensitivity / eps
        noise = np.random.laplace(0, scale)
        return value + noise
    
    if st.button("Apply Differential Privacy"):
        # Set sensitivity
        sensitivity = np.percentile(income_data, 99) - np.percentile(income_data, 1)
        
        # Apply DP to statistics
        private_mean = add_laplace_noise(income_data.mean(), sensitivity/len(income_data), epsilon)
        private_max = add_laplace_noise(income_data.max(), sensitivity, epsilon*0.1)
        
        # Display differentially private statistics
        st.subheader("Differentially Private Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Income (Private)", f"${private_mean:.2f}", 
                     delta=f"{private_mean - income_data.mean():.2f}")
        with col2:
            st.metric("Max Income (Private)", f"${private_max:.2f}", 
                     delta=f"{private_max - income_data.max():.2f}")
        
        # Visualize the effect of DP on the histogram
        st.subheader("Income Distribution")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Original histogram
        ax1.hist(income_data, bins=20, alpha=0.7, color='blue')
        ax1.set_title('Original Distribution')
        ax1.set_xlabel('Income')
        
        # DP histogram
        hist, bin_edges = np.histogram(income_data, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        noisy_hist = np.array([add_laplace_noise(count, 2, epsilon) for count in hist])
        noisy_hist = np.maximum(noisy_hist, 0)  # Ensure counts are non-negative
        
        ax2.bar(bin_centers, noisy_hist, width=(bin_edges[1]-bin_edges[0])*0.8, alpha=0.7, color='green')
        ax2.set_title('Private Distribution')
        ax2.set_xlabel('Income')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info(f"""
        The differentially private results have noise added to protect individual privacy.
        With ε={epsilon}, {'strong' if epsilon < 0.1 else 'moderate' if epsilon < 1.0 else 'basic'} 
        privacy guarantees are provided.
        """)

# Federated Learning Demo
elif page == "Federated Learning":
    st.markdown("## Federated Learning Simulation")
    
    st.markdown("""
    
    Federated Learning allows multiple parties to train a shared model without sharing their raw data.
    Each device trains on local data and only shares model updates, not the data itself.
    
   """)
    
    # Federated Learning Simulation
    st.subheader("Interactive Federated Learning Simulation")
    
    # Create simulated data for different clients
    np.random.seed(42)
    
    # Function to generate client data with different distributions
    def generate_client_data(center_x, center_y, n_samples=100):
        x = np.random.normal(center_x, 1.5, n_samples)
        y = np.random.normal(center_y, 1.5, n_samples)
        return np.vstack((x, y)).T
    
    # Generate data for 4 clients
    client1_data = generate_client_data(3, 3, 100)
    client2_data = generate_client_data(-3, 3, 100)
    client3_data = generate_client_data(-3, -3, 100)
    client4_data = generate_client_data(3, -3, 100)
    
    # Visualization of client data
    st.write("### Client Data Distribution")
    st.write("Each client has their own private dataset that they don't share.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(client1_data[:, 0], client1_data[:, 1], alpha=0.6, label="Client 1")
    ax.scatter(client2_data[:, 0], client2_data[:, 1], alpha=0.6, label="Client 2")
    ax.scatter(client3_data[:, 0], client3_data[:, 1], alpha=0.6, label="Client 3") 
    ax.scatter(client4_data[:, 0], client4_data[:, 1], alpha=0.6, label="Client 4")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.set_title("Private Data on Each Client")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Federated learning process
    st.subheader("Federated Learning Process")
    
    # Simulation parameters
    n_rounds = st.slider("Number of Federated Training Rounds", 1, 10, 3)
    participating_clients = st.multiselect(
        "Select participating clients",
        ["Client 1", "Client 2", "Client 3", "Client 4"],
        default=["Client 1", "Client 2", "Client 3", "Client 4"]
    )
    
    if st.button("Run Federated Learning Simulation"):
        st.write("### Federated Learning Simulation")
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simple visualization of model improvement
        fig, axes = plt.subplots(1, n_rounds, figsize=(4*n_rounds, 4))
        if n_rounds == 1:
            axes = [axes]  # Make sure axes is iterable
            
        # Initial model (simple linear boundary for visualization)
        global_model = {"w1": 0.0, "w2": 0.0, "b": 0.0}
        
        # Simulation of federated learning rounds
        for round_idx in range(n_rounds):
            status_text.text(f"Round {round_idx+1}/{n_rounds}: Training local models...")
            
            # Simulate local training on each device
            local_updates = []
            
            # For each participating client, compute a local update
            client_data_mapping = {
                "Client 1": client1_data,
                "Client 2": client2_data,
                "Client 3": client3_data,
                "Client 4": client4_data
            }
            
            for client in participating_clients:
                client_data = client_data_mapping[client]
                
                # Simulate local model improvement (in a real system, this would be actual training)
                # Here we're just creating a simplified linear model update
                # These are just made-up examples to visualize the concept
                if client == "Client 1":
                    local_update = {"w1": 0.2, "w2": 0.1, "b": 0.05}
                elif client == "Client 2":
                    local_update = {"w1": -0.1, "w2": 0.2, "b": -0.05}
                elif client == "Client 3":
                    local_update = {"w1": -0.2, "w2": -0.1, "b": 0.1}
                else:  # Client 4
                    local_update = {"w1": 0.1, "w2": -0.2, "b": -0.1}
                
                # Add some randomness to make it interesting
                local_update["w1"] += np.random.normal(0, 0.05)
                local_update["w2"] += np.random.normal(0, 0.05)
                local_update["b"] += np.random.normal(0, 0.02)
                
                local_updates.append(local_update)
            
            # Aggregate updates (simple averaging for this demo)
            if local_updates:
                avg_w1 = sum(update["w1"] for update in local_updates) / len(local_updates)
                avg_w2 = sum(update["w2"] for update in local_updates) / len(local_updates)
                avg_b = sum(update["b"] for update in local_updates) / len(local_updates)
                
                # Update global model
                global_model["w1"] += avg_w1
                global_model["w2"] += avg_w2
                global_model["b"] += avg_b
            
            status_text.text(f"Round {round_idx+1}/{n_rounds}: Aggregating model updates...")
            
            # Visualize the current model
            ax = axes[round_idx]
            
            # Plot all client data
            all_clients_data = np.vstack((client1_data, client2_data, client3_data, client4_data))
            ax.scatter(all_clients_data[:, 0], all_clients_data[:, 1], c='lightgray', alpha=0.3)
            
            # Plot participating clients with proper colors
            client_colors = {'Client 1': 'blue', 'Client 2': 'orange', 'Client 3': 'green', 'Client 4': 'red'}
            for client in participating_clients:
                client_data = client_data_mapping[client]
                ax.scatter(client_data[:, 0], client_data[:, 1], alpha=0.6, label=client, color=client_colors[client])
            
            # Plot decision boundary (simplified for visualization)
            x_min, x_max = all_clients_data[:, 0].min() - 1, all_clients_data[:, 0].max() + 1
            y_min, y_max = all_clients_data[:, 1].min() - 1, all_clients_data[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            
            # Simple linear decision boundary: w1*x + w2*y + b = 0
            Z = global_model["w1"] * xx + global_model["w2"] * yy + global_model["b"]
            ax.contour(xx, yy, Z, levels=[0], colors='k', linestyles='--')
            
            ax.set_title(f"Round {round_idx+1}")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.grid(True, alpha=0.3)
            
            # Update progress
            progress_bar.progress((round_idx + 1) / n_rounds)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        status_text.text("Federated learning simulation complete!")
        

# Homomorphic Encryption Demo
elif page == "Homomorphic Encryption":
    st.markdown("## Homomorphic Encryption")
    
    st.markdown("""
    
    Homomorphic encryption allows computations to be performed on encrypted data without decrypting it.
    The results, when decrypted, match the results of the same operations performed on the plaintext.
    
    """)
    
    # Simple Homomorphic Encryption Example
    st.subheader("Interactive Homomorphic Encryption Demo")
    
    # Simple encryption function (for demonstration only - not secure!)
    def simple_encrypt(value, key):
        # This is a very simple "encryption" for demo purposes only
        # NOT suitable for actual security use!
        return (value * key) % 10000
    
    def simple_decrypt(encrypted_value, key):
        # Simple corresponding "decryption"
        inv_key = pow(key, -1, 10000)  # Modular multiplicative inverse
        return (encrypted_value * inv_key) % 10000
    
    # Create a simple interactive demo
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Owner Side")
        
        # User inputs
        value1 = st.number_input("Enter first value", min_value=1, max_value=100, value=42)
        value2 = st.number_input("Enter second value", min_value=1, max_value=100, value=57)
        
        # Secret key (normally would be securely generated)
        encryption_key = st.number_input("Encryption key (normally private)", min_value=1, max_value=9999, value=123)
        
        # Encrypt values
        if st.button("Encrypt Values"):
            encrypted_value1 = simple_encrypt(value1, encryption_key)
            encrypted_value2 = simple_encrypt(value2, encryption_key)
            
            st.session_state.encrypted_value1 = encrypted_value1
            st.session_state.encrypted_value2 = encrypted_value2
            st.session_state.encryption_key = encryption_key
            
            st.write(f"Encrypted Value 1: {encrypted_value1}")
            st.write(f"Encrypted Value 2: {encrypted_value2}")
            
            st.success("Values encrypted! Cloud service can now perform computations on encrypted data.")
    
    with col2:
        st.subheader("Cloud Service Side")
        
        if 'encrypted_value1' in st.session_state and 'encrypted_value2' in st.session_state:
            st.write(f"Received Encrypted Value 1: {st.session_state.encrypted_value1}")
            st.write(f"Received Encrypted Value 2: {st.session_state.encrypted_value2}")
            
            operation = st.selectbox("Select operation to perform on encrypted data", ["Addition", "Multiplication"])
            
            if st.button("Perform Computation on Encrypted Data"):
                if operation == "Addition":
                    # For the simple encryption scheme we're using, to perform homomorphic addition:
                    # We need special operations for this specific encryption method
                    # In real FHE, this would be handled by the encryption scheme
                    encrypted_result = (st.session_state.encrypted_value1 + st.session_state.encrypted_value2) % 10000
                    real_operation = "addition"
                else:  # Multiplication
                    # Again, this is a simplified example
                    encrypted_result = (st.session_state.encrypted_value1 * st.session_state.encrypted_value2) % 10000
                    real_operation = "multiplication"
                
                st.session_state.encrypted_result = encrypted_result
                st.session_state.operation = real_operation
                
                st.write(f"Encrypted Result: {encrypted_result}")
                st.info("The cloud service performed the computation without decrypting the data!")
        else:
            st.info("Waiting for encrypted values from the data owner...")
    
    # Result section
    if 'encrypted_result' in st.session_state:
        st.subheader("Result Decryption (Data Owner Side)")
        
        if st.button("Decrypt Result"):
            # Decrypt the result
            # For this simplified demo, the decryption is different based on operation
            if st.session_state.operation == "addition":
                decrypted_result = simple_decrypt(st.session_state.encrypted_result, st.session_state.encryption_key)
            else:  # multiplication
                # For multiplication, in this simple scheme, we need to handle it differently
                # This is a major simplification compared to real homomorphic encryption
                decrypted_result = simple_decrypt(st.session_state.encrypted_result, 
                                              (st.session_state.encryption_key ** 2) % 10000)
            
            # Calculate expected result for verification
            if st.session_state.operation == "addition":
                expected_result = value1 + value2
            else:
                expected_result = value1 * value2
            
            st.write(f"Decrypted Result: {decrypted_result}")
            st.write(f"Expected Result (for verification): {expected_result}")
            
            st.success(f"""
            The cloud service computed the {st.session_state.operation} of your values without ever seeing the actual numbers!
            
            This simple demo illustrates the concept of homomorphic encryption. 
            Real homomorphic encryption schemes are much more complex and secure.
            """)
    