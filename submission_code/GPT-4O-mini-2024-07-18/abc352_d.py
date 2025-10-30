def min_good_index_sequence(N, K, P):
    from collections import defaultdict
    
    # Dictionary to store the positions of each number in P
    position = defaultdict(list)
    
    for index in range(N):
        position[P[index]].append(index + 1)  # Store 1-based index
    
    min_length = float('inf')
    
    # Check for each starting number a from 1 to N-K+1
    for a in range(1, N - K + 2):
        # We need to find the indices of the numbers a, a+1, ..., a+K-1
        indices = []
        for num in range(a, a + K):
            if num in position:
                indices.extend(position[num])
        
        if len(indices) >= K:
            # Sort the indices to find the minimum range
            indices.sort()
            # We only need the first K indices to find the range
            current_length = indices[K - 1] - indices[0]
            min_length = min(min_length, current_length)
    
    return min_length

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
P = list(map(int, data[2:]))

# Get the result
result = min_good_index_sequence(N, K, P)

# Print the result
print(result)