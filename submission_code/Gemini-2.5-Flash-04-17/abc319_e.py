import sys

# Function to calculate earliest arrival time at Aoki's house
# when leaving Takahashi's house at 'start_time'.
def calculate_arrival_for_base(start_time, N, X, Y, bus_info):
    """
    Simulates the journey starting at start_time.

    Args:
        start_time: The departure time from Takahashi's house.
        N: The number of bus stops (1 to N).
        X: Walking time from house to stop 1.
        Y: Walking time from stop N to Aoki's house.
        bus_info: List of (P_i, T_i) pairs for bus i from stop i to stop i+1.
                  bus_info[0] is (P_1, T_1), bus_info[i] is (P_{i+1}, T_{i+1}).

    Returns:
        The earliest arrival time at Aoki's house.
    """
    
    # Arrive at bus stop 1
    current_time = start_time + X
    
    # Simulate bus travel through N-1 bus routes (from stop 1 to 2, ..., from stop N-1 to N)
    # The input gives P_i, T_i for the bus from stop i to stop i+1, for i = 1, ..., N-1.
    # The bus_info list stores these pairs sequentially:
    # bus_info[0] contains (P_1, T_1) for the bus from stop 1 to stop 2.
    # bus_info[1] contains (P_2, T_2) for the bus from stop 2 to stop 3.
    # ...
    # bus_info[i] contains (P_{i+1}, T_{i+1}) for the bus from stop i+1 to stop i+2.
    
    # The loop iterates through the N-1 bus segments.
    # In the k-th iteration (0-indexed, k from 0 to N-2), we process the bus
    # from stop k+1 to stop k+2 using bus_info[k] = (P_{k+1}, T_{k+1}).
    # At the start of the k-th iteration, current_time is the arrival time at stop k+1.
    
    for i in range(N - 1):
        # Get (P, T) for the bus from stop i+1 to stop i+2
        P, T = bus_info[i] 
        
        # current_time is the arrival time at bus stop i+1
        
        # Calculate the earliest departure time from bus stop i+1
        # It must be a multiple of P (P_{i+1}) and >= current_time
        # (current_time + P - 1) // P * P calculates the smallest multiple of P >= current_time
        departure_time = (current_time + P - 1) // P * P
        
        # Calculate arrival time at bus stop i+2
        arrival_next_stop = departure_time + T
        
        # Update current time to be the arrival time at bus stop i+2
        current_time = arrival_next_stop
        
    # After the loop (i = N-2), current_time holds the arrival time at bus stop (N-2)+2 = N.
    arrival_at_bus_stop_N = current_time
    
    # Calculate the final arrival time at Aoki's house
    final_arrival_time = arrival_at_bus_stop_N + Y
    
    return final_arrival_time

# --- Main Execution ---

# Read N, X, Y
N, X, Y = map(int, sys.stdin.readline().split())

# Read P_i, T_i for i = 1 to N-1
# Store them such that bus_info[i] corresponds to the i-th input pair (0-indexed).
# So bus_info[0] = (P_1, T_1), bus_info[1] = (P_2, T_2), ..., bus_info[N-2] = (P_{N-1}, T_{N-1}).
bus_info = []
for _ in range(N - 1):
    P, T = map(int, sys.stdin.readline().split())
    bus_info.append((P, T))

# Read Q
Q = int(sys.stdin.readline())

# The LCM of numbers from 1 to 8 is 840.
# Let H(q) be the earliest arrival time at Aoki's house when leaving Takahashi's house at time q.
# Due to the nature of bus schedules (departures at multiples of P_i, where P_i divides LCM),
# the transformation from arrival time at stop i to arrival time at stop i+1, modulo LCM,
# depends only on the arrival time at stop i modulo LCM.
# This implies that H(q + k * LCM) = H(q) + k * LCM for any integer k.
# We can leverage this property by precomputing H(r) for r = 0, 1, ..., LCM - 1.
L = 840 

# Precompute H(r) for r = 0, ..., L-1
precomputed_results = [0] * L

for start_rem in range(L):
    # Calculate H(start_rem) by simulating the journey starting exactly at time `start_rem`
    precomputed_results[start_rem] = calculate_arrival_for_base(start_rem, N, X, Y, bus_info)

# Process queries
# For each query q_i, calculate the answer using the precomputed value for q_i % L
for _ in range(Q):
    q = int(sys.stdin.readline())
    
    # Express q as q = k * L + r, where r = q % L (0 <= r < L)
    k = q // L
    r = q % L
    
    # The answer for q is H(q) = H(k * L + r) = H(r) + k * L
    ans = precomputed_results[r] + k * L
    print(ans)