def min_total_dissatisfaction(N, K, X, T):
    # Initialize variables
    total_dissatisfaction = 0
    current_day = 1  # Start shipping from Day 1
    i = 0  # Index for orders

    while i < N:
        # Determine the day we can ship the next batch
        current_day = max(current_day, T[i])  # We can't ship before the order is placed
        
        # Ship at most K orders
        for j in range(K):
            if i < N and T[i] <= current_day:
                total_dissatisfaction += (current_day - T[i])
                i += 1  # Move to the next order
            else:
                break  # No more orders to ship in this batch
        
        # Move to the next possible shipping day
        current_day += X

    return total_dissatisfaction

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
X = int(data[2])
T = list(map(int, data[3:]))

# Calculate and print the result
result = min_total_dissatisfaction(N, K, X, T)
print(result)