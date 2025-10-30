def count_triples(N, A):
    # Create a dictionary to store the indices of each element
    indices = {}
    for i, a in enumerate(A):
        if a not in indices:
            indices[a] = []
        indices[a].append(i)
    
    # Count the number of valid triples
    count = 0
    for i in range(N):
        for k in range(i+2, N):
            if A[i] == A[k] and A[i] != A[j]:
                # Find the number of j's such that i < j < k
                # Since indices are sorted, we can use binary search
                # to find the first index greater than i and the last index less than k
                left = bisect.bisect_right(indices[A[i]], i)
                right = bisect.bisect_left(indices[A[i]], k)
                count += right - left
    
    return count

import bisect
import sys

# Read input
N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

# Solve and output
print(count_triples(N, A))