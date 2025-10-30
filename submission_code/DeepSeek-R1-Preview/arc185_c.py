import bisect
from collections import defaultdict

def find_triple():
    # Read input
    n, x = map(int, input().split())
    a = list(map(int, input().split()))
    
    # Create a dictionary to map each value to its list of indices
    value_indices = defaultdict(list)
    for idx, num in enumerate(a):
        value_indices[num].append(idx)
    
    # Determine the maximum i to check
    max_i = min(999, n - 3)
    for i in range(max_i + 1):
        # Determine the range for j
        start_j = i + 1
        end_j = min(i + 1000, n - 2)  # j can be at most n-2 since k needs to be j+1 <= n-1
        for j in range(start_j, end_j + 1):
            target = x - a[i] - a[j]
            if target < 1:
                continue  # since all elements are positive
            if target in value_indices:
                indices = value_indices[target]
                # Find the first index k > j
                pos = bisect.bisect_right(indices, j)
                if pos < len(indices):
                    k = indices[pos]
                    print(i + 1, j + 1, k + 1)
                    return
    # If no triple found
    print(-1)

find_triple()