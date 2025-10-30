def has_multiple_subsequences(N, M, A, B):
    from collections import defaultdict
    
    # Create a mapping of indices for elements in A
    index_map = defaultdict(list)
    for i in range(N):
        index_map[A[i]].append(i)
    
    # To store the last index used for matching B
    last_index = -1
    count = 0
    
    for b in B:
        if b not in index_map:
            return "No"
        
        # Get the list of indices for the current element in B
        indices = index_map[b]
        
        # Find the first index in indices that is greater than last_index
        found = False
        for idx in indices:
            if idx > last_index:
                last_index = idx
                found = True
                break
        
        if not found:
            return "No"
        
        # Count how many subsequences can be formed
        count += len(indices) - indices.index(last_index) - 1
    
    return "Yes" if count > 0 else "No"

import sys
input = sys.stdin.read
data = input().splitlines()

# Read N and M
N, M = map(int, data[0].split())
# Read A and B
A = list(map(int, data[1].split()))
B = list(map(int, data[2].split()))

# Get the result
result = has_multiple_subsequences(N, M, A, B)
print(result)