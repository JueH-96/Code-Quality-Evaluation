# YOUR CODE HERE
import heapq
from collections import Counter

N, K, Q = map(int, input().split())

# Initialize array A with all zeros
A = [0] * (N + 1)  # 1-indexed

# For efficient computation of top K elements, we'll maintain a sorted list of values
# We'll use a Counter to track frequency of each value
value_count = Counter()
value_count[0] = N

for _ in range(Q):
    X, Y = map(int, input().split())
    
    # Remove old value from counter
    old_value = A[X]
    value_count[old_value] -= 1
    if value_count[old_value] == 0:
        del value_count[old_value]
    
    # Add new value to counter
    A[X] = Y
    value_count[Y] += 1
    
    # Calculate sum of K largest elements
    # Get all unique values in descending order
    values = sorted(value_count.keys(), reverse=True)
    
    total = 0
    remaining = K
    
    for val in values:
        count = value_count[val]
        take = min(remaining, count)
        total += val * take
        remaining -= take
        if remaining == 0:
            break
    
    print(total)