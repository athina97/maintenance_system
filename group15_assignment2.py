import numpy as np 
import streamlit as st 
import matplotlib.pyplot as plt



def calculate_transition_rates(n, k, failure_rate, repair_rate, standby_mode, num_repairmen):
    num_states = n + 1
    Q = np.zeros((num_states, num_states))
    
    for i in range(num_states):
        if i > 0:
            if standby_mode == "Cold":
                Q[i, i-1] = min(i,k)* failure_rate
            elif standby_mode == "Warm":
                Q[i, i-1] = i * failure_rate
        
        if i < n:
            failed_components = n - i
            repairs = min(failed_components, num_repairmen)
            Q[i, i+1] = repairs * repair_rate
    
    for i in range(num_states):
        Q[i, i] = -np.sum(Q[i, :])
    
    return Q

def compute_stationary_distribution(Q):
    num_states = Q.shape[0]
    A = np.vstack([Q.T, np.ones(num_states)])
    b = np.hstack([np.zeros(num_states), 1])
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

def calculate_uptime_fraction(pi, k):
    return np.sum(pi[k:])

def calculate_total_cost(n, num_repairmen, cost_per_component, cost_per_repairman, downtime_cost, uptime_fraction):
    return (n * cost_per_component + 
            num_repairmen * cost_per_repairman + 
            (1 - uptime_fraction) * downtime_cost)

def optimize_system(max_n, max_repairmen, k, failure_rate, repair_rate, standby_mode, 
                   cost_per_component, cost_per_repairman, downtime_cost):
    min_cost = float('inf')
    optimal_n = None
    optimal_repairmen = None
   
    
    for n in range(k, max_n + 1):
        for num_repairmen in range(1, max_repairmen + 1):
            Q = calculate_transition_rates(n, k, failure_rate, repair_rate, standby_mode, num_repairmen)
            pi = compute_stationary_distribution(Q)
            uptime_fraction = calculate_uptime_fraction(pi, k)
            total_cost = calculate_total_cost(n, num_repairmen, cost_per_component, cost_per_repairman, downtime_cost, uptime_fraction)
            
            
            if total_cost < min_cost:
                min_cost = total_cost
                optimal_n = n
                optimal_repairmen = num_repairmen
    
    return optimal_n, optimal_repairmen, min_cost

st.title("k-out-of-n Maintenance System")

mode = st.sidebar.radio("Select Mode:", ["System Availability", "Cost Optimization"])

if mode == "System Availability":
    st.sidebar.header("System Parameters")
    n = st.sidebar.number_input("Number of components (n):", min_value=1, value=3)
    k = st.sidebar.number_input("Number of components needed to function (k):", min_value=1, max_value=n, value=2)
    failure_rate = st.sidebar.number_input("Failure rate (λ):", min_value=0.0, value=1.0)
    repair_rate = st.sidebar.number_input("Repair rate (μ):", min_value=0.0, value=1.0)
    standby_mode = st.sidebar.selectbox("Stand-by mode:", ["Cold", "Warm"])
    num_repairmen = st.sidebar.number_input("Number of repairmen:", min_value=1, value=1)
    
    # calculations
    Q = calculate_transition_rates(n, k, failure_rate, repair_rate, standby_mode, num_repairmen)
    pi = compute_stationary_distribution(Q)
    uptime_fraction = calculate_uptime_fraction(pi, k)
    
    
    st.subheader("Stationary Distribution (Plot)")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(range(n+1), pi, color='skyblue', edgecolor='black')
    ax.set_xlabel("Number of Working Components")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(n+1))  # Label every state
    st.pyplot(fig)
    
    st.subheader("Stationary Distribution (Table)")
    st.dataframe({
        "State": [f"{i} components up" for i in range(n+1)],
        "Probability": pi
    })
    st.subheader("System Availability")
    st.info(f"The fraction of time the system is up: {uptime_fraction:.4f}")
    

else:
    st.sidebar.header("System Parameters")
    max_n = st.sidebar.number_input("Maximum number of components to test (n):", min_value=1, value=10)
    max_repairmen = st.sidebar.number_input("Maximum number of repairmen to test:", min_value=1, value=5)
    k = st.sidebar.number_input("Number of components needed to function (k):", min_value=1, value=2)
    failure_rate = st.sidebar.number_input("Failure rate (λ):", min_value=0.0, value=1.0)
    repair_rate = st.sidebar.number_input("Repair rate (μ):", min_value=0.0, value=0.5)
    standby_mode = st.sidebar.selectbox("Stand-by mode:", ["Cold","Warm"])
    
    
    st.sidebar.header("Cost Parameters")
    cost_per_component = st.sidebar.number_input("Cost per component:", min_value=0.0, value=50.0)
    cost_per_repairman = st.sidebar.number_input("Cost per repairman:", min_value=0.0, value=50.0)
    downtime_cost = st.sidebar.number_input("Downtime cost per unit time:", min_value=0.0, value=1000.0)
    
    
    if st.sidebar.button("Run Optimization"):
        with st.spinner("Finding optimal configuration..."):
            optimal_n, optimal_repairmen, min_cost = optimize_system(
                max_n, max_repairmen, k, failure_rate, repair_rate, 
                standby_mode, cost_per_component, cost_per_repairman, downtime_cost
            )
        
        st.header("Optimization Results")
        st.info(f"Optimal number of components (n): {optimal_n}")
        st.info(f"Optimal number of repairmen: {optimal_repairmen}")
        st.info(f"Minimum total cost: {min_cost:.2f}")
        
    
