import sys
import bisect

# Read all input from stdin
data = sys.stdin.read().split()
index = 0

# Read N and Q
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Read and sort the A coordinates
A = list(map(int, data[index:index + N]))
index += N
A.sort()

# Process each query
for _ in range(Q):
    # Read b and k for the query
    b = int(data[index])
    k = int(data[index + 1])
    index += 2
    
    # Binary search for the smallest distance D such that at least k points are within distance D
    low = 0
    high = 200000000  # Maximum possible distance is 2e8
    
    while low < high:
        mid = (low + high) // 2
        left_val = b - mid
        right_val = b + mid
        left_idx = bisect.bisect_left(A, left_val)
        right_idx = bisect.bisect_right(A, right_val)
        count = right_idx - left_idx
        
        if count >= k:
            high = mid
        else:
            low = mid + 1
    
    # low is the k-th smallest distance
    print(low)