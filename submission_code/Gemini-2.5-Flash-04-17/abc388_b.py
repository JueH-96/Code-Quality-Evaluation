import sys

# Read N and D from the first line
line1 = sys.stdin.readline().split()
N = int(line1[0])
D = int(line1[1])

# Read snake data (thickness T and initial length L)
snakes = []
for _ in range(N):
    line = sys.stdin.readline().split()
    T = int(line[0])
    L = int(line[1])
    snakes.append((T, L))

# Iterate for each possible length increment k, from 1 up to D
for k in range(1, D + 1):
    # Initialize the maximum weight found so far for the current increment k
    # Since all T_i and L_i are at least 1, initial lengths (L_i + k) will
    # also be at least 1+1 = 2 (when k=1, L_i=1), and thickness T_i >= 1.
    # Thus, weights (T_i * (L_i + k)) will always be positive.
    # Initializing with 0 is a safe lower bound for the maximum weight.
    max_weight_k = 0
    
    # Iterate through each snake to calculate its weight with the current increment k
    for T, L in snakes:
        # Calculate the current snake's weight: Thickness * (Initial Length + Increment k)
        current_weight = T * (L + k)
        
        # Update the maximum weight found for this increment k if the current snake is heavier
        if current_weight > max_weight_k:
            max_weight_k = current_weight
            
    # After checking all snakes for the current increment k, print the maximum weight found.
    print(max_weight_k)