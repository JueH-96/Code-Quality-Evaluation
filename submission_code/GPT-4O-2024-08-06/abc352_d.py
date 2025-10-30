def find_minimum_good_sequence_distance(N, K, P):
    min_distance = float('inf')
    
    # Use a sliding window of size K
    for start in range(N - K + 1):
        end = start + K
        window = P[start:end]
        min_val = min(window)
        max_val = max(window)
        
        # Check if the window can be rearranged to form consecutive integers
        if max_val - min_val + 1 == K:
            min_distance = min(min_distance, end - start - 1)
    
    return min_distance

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
P = list(map(int, data[2:]))

# Output the result
print(find_minimum_good_sequence_distance(N, K, P))