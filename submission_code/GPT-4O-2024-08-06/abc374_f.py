def minimize_dissatisfaction(N, K, X, T):
    # Initialize variables
    total_dissatisfaction = 0
    current_day = T[0]  # Start from the day of the first order

    i = 0
    while i < N:
        # Determine the earliest day we can ship the next batch
        current_day = max(current_day, T[i])
        
        # Ship up to K orders starting from the current order
        batch_size = min(K, N - i)
        for j in range(batch_size):
            order_index = i + j
            # Calculate dissatisfaction for this order
            dissatisfaction = current_day - T[order_index]
            total_dissatisfaction += dissatisfaction
        
        # Move to the next batch
        i += batch_size
        # Update the next available shipping day
        current_day += X

    return total_dissatisfaction

# Reading input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
K = int(data[1])
X = int(data[2])
T = list(map(int, data[3:]))

# Calculate and print the result
result = minimize_dissatisfaction(N, K, X, T)
print(result)