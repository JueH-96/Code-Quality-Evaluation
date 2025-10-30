# YOUR CODE HERE
import sys

# Function to calculate the next departure time
# Given arrival_time at a bus stop and bus frequency P
# The bus departs at exact multiples of P. If arrival_time is a multiple,
# Takahashi can take that bus immediately. Otherwise, he waits for the next multiple.
# Example: arrival_time = 13, P = 5 -> next_multiple = 15
# Example: arrival_time = 15, P = 5 -> next_multiple = 15 (no wait)
# Example: arrival_time = 0, P = 5 -> next_multiple = 0 (no wait)
def get_next_departure_time(arrival_time, P):
    # This formula correctly calculates the smallest multiple of P that is >= arrival_time.
    # It works for arrival_time >= 0 and P >= 1.
    return ((arrival_time + P - 1) // P) * P

def solve():
    # Read N, X, Y from the first line
    N, X, Y = map(int, sys.stdin.readline().split())

    # Read P_i and T_i for each bus segment.
    # bus_info[j] contains (P_{j+1}, T_{j+1}) for the bus from stop j+1 to j+2.
    # There are N-1 such segments.
    bus_info = []
    for _ in range(N - 1):
        P_i, T_i = map(int, sys.stdin.readline().split())
        bus_info.append((P_i, T_i))

    # Read Q, the number of queries
    Q = int(sys.stdin.readline())
    queries = []
    # Read each query starting time q_i
    for _ in range(Q):
        queries.append(int(sys.stdin.readline()))

    # Calculate the Least Common Multiple (LCM) of all possible P_i values (1 to 8).
    # The P_i values are constrained to be between 1 and 8.
    # LCM(1, 2, 3, 4, 5, 6, 7, 8) = 2^3 * 3 * 5 * 7 = 8 * 3 * 5 * 7 = 840.
    # This LCM value (840) is crucial because it ensures the travel time characteristics
    # from Bus Stop 1 onwards are periodic.
    LCM_val = 840 

    # Precomputation array: h[k] will store the total time taken to travel
    # from Bus Stop 1 to Aoki's house, assuming one *arrives* at Bus Stop 1 at time `k`.
    # More precisely, if one arrives at Bus Stop 1 at time `t`, the arrival at Aoki's house
    # will be `t + h[t % LCM_val]`.
    # `h[k]` essentially captures the sum of all T_i values, Y, and all waiting times,
    # given an arrival time `k` at Bus Stop 1.
    h = [0] * LCM_val

    # Precompute for all possible remainder values modulo LCM_val (0 to LCM_val - 1)
    for k in range(LCM_val):
        # Simulate travel starting from Bus Stop 1, assuming arrival at time `k`.
        current_time_at_bus_stop = k
        
        # Traverse through all bus segments from Bus Stop 1 to Bus Stop N.
        # `bus_info` contains the P_i, T_i pairs in order.
        for P_i, T_i in bus_info:
            # Calculate the time at which the bus departs from the current bus stop.
            departure_time = get_next_departure_time(current_time_at_bus_stop, P_i)
            # Calculate the arrival time at the next bus stop.
            current_time_at_bus_stop = departure_time + T_i
        
        # Add the final walking time from Bus Stop N to Aoki's house.
        final_arrival_time_simulated = current_time_at_bus_stop + Y
        
        # Store the total elapsed time from the point of arrival at Bus Stop 1 (`k`)
        # until the final arrival at Aoki's house. This is `(total_time_spent_in_bus_system_plus_final_walk)`.
        # `h[k]` represents `(arrival_time_at_Aoki's_house - arrival_time_at_BS1)` for this simulation.
        h[k] = final_arrival_time_simulated - k

    # Process each query using the precomputed `h` array
    results = []
    for q_i in queries:
        # Calculate the time Takahashi arrives at Bus Stop 1.
        # This is the initial departure time from house plus the walking time X.
        time_at_BS1 = q_i + X
        
        # Determine the effective "starting point" for the bus system simulation
        # by taking the modulo with LCM_val.
        remainder_for_lookup = time_at_BS1 % LCM_val
        
        # Retrieve the precomputed offset (total travel time from BS1 to Aoki's)
        # for this specific remainder.
        offset_from_precomputation = h[remainder_for_lookup]
        
        # The total arrival time at Aoki's house is the arrival time at BS1
        # plus the total time taken for the rest of the journey (from BS1 to Aoki's).
        total_arrival_time = time_at_BS1 + offset_from_precomputation
        results.append(total_arrival_time)

    # Print all computed results, each on a new line
    for res in results:
        print(res)

solve()