# YOUR CODE HERE
import sys
import math

def calculate_lcm(a, b):
    """
    Calculates the least common multiple of a and b. 
    Assumes a, b >= 1 based on problem constraints P_i >= 1.
    """
    # The formula lcm(a, b) = (a * b) // gcd(a, b) is used.
    # Since a, b >= 1 are guaranteed by constraints, math.gcd(a, b) >= 1, 
    # ensuring no division by zero. Python's arbitrary precision integers handle large products.
    # However, intermediate LCM values and final LCM will not exceed 840 based on P_i <= 8.
    if a == 0 or b == 0: # Should not happen with P_i >= 1 but defensive check
        return 0
    return (a * b) // math.gcd(a, b)

def solve():
    # Read problem parameters N, X, Y
    N, X, Y = map(int, sys.stdin.readline().split())
    
    buses = []
    # Read N-1 bus route parameters (P_i, T_i)
    for _ in range(N - 1):
        # Store as tuple for potentially better performance/immutability
        buses.append(tuple(map(int, sys.stdin.readline().split())))

    # Calculate the least common multiple (LCM) of all bus periods P_i.
    # This LCM, L_actual, is the period of the total journey time difference function.
    # Initialize LCM to 1, which is the identity element for LCM.
    L_actual = 1
    if N > 1: # If N=1, there are no buses, L_actual remains 1. But constraint N>=2 holds.
       # Collect unique P values to avoid redundant LCM calculations.
       # set() automatically handles duplicates.
       unique_P = set(bus[0] for bus in buses)
       
       # Iteratively compute the LCM of all unique P values.
       current_lcm = 1
       for p in unique_P:
           # Since P_i >= 1, all values 'p' are >= 1.
           # The current_lcm will also remain >= 1.
           current_lcm = calculate_lcm(current_lcm, p)
       L_actual = current_lcm
    
    # L_actual is guaranteed to be >= 1. Its maximum value is lcm(1, 2, ..., 8) = 840.

    # Precompute the journey time difference: Arrival_Time(k) - Start_Time(k) for k = 0..L_actual-1
    # This difference, let's call it D(k), has the property D(k) = D(k mod L_actual).
    # The precomputation step has time complexity O(N * L_actual).
    journey_time_diff = [0] * L_actual

    for k in range(L_actual):
        # Simulate the journey starting exactly at time k from Takahashi's house.
        
        # Arrive at bus stop 1 after walking X time units.
        current_time = k + X 
        
        # Simulate travel through the N-1 bus segments from bus stop 1 to bus stop N.
        for i in range(N - 1):
            P, T = buses[i] # Get period P_i and travel time T_i for segment i -> i+1
            
            # Calculate the departure time from the current bus stop (stop i+1).
            # Need to wait until the next multiple of P_i.
            # Departure time is the smallest multiple of P that is >= current_time.
            # This can be calculated using ceiling division: ceil(current_time / P) * P
            # In integer arithmetic, this is equivalent to: ((current_time + P - 1) // P) * P
            departure_time = ((current_time + P - 1) // P) * P
            
            # Arrive at the next bus stop (stop i+2) after traveling T_i time units.
            current_time = departure_time + T 
            
        # After arriving at bus stop N (at current_time), walk to Aoki's house.
        # This takes Y time units.
        final_time = current_time + Y
        
        # Store the total time elapsed for the journey relative to the start time k.
        # This difference value will be used for queries.
        journey_time_diff[k] = final_time - k

    # Read the number of queries Q.
    Q = int(sys.stdin.readline())
    
    # Process each query. This part has time complexity O(Q).
    results = []
    for _ in range(Q):
        # Read the start time q_i for the current query.
        q_i = int(sys.stdin.readline()) 
        
        # The journey time difference depends only on q_i modulo L_actual.
        # Use the property A(q) - q = D(q) = D(q mod L), where A(q) is arrival time for start q.
        remainder = q_i % L_actual
        
        # Retrieve the precomputed journey time difference for this remainder class.
        diff = journey_time_diff[remainder]
        
        # Calculate the final arrival time: Start time q_i + Time Difference 'diff'.
        # A(q_i) = q_i + D(q_i) = q_i + D(q_i mod L_actual)
        arrival_time = q_i + diff
        results.append(str(arrival_time))

    # Print all computed arrival times, each on a new line.
    # Using sys.stdout.write and join is generally faster for large number of outputs.
    sys.stdout.write("
".join(results) + "
")

# Execute the main solver function.
solve()