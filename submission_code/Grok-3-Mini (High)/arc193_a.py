import sys
import bisect
import math

# Read all input
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read W
W = list(map(int, data[index:index + N]))
index += N

# Read L and R for each vertex
L = [0] * N
R = [0] * N
for i in range(N):
    L[i] = int(data[index])
    index += 1
    R[i] = int(data[index])
    index += 1

# Read Q
Q = int(data[index])
index += 1

# Read Q queries, convert to 0-based indexing
queries = []
for _ in range(Q):
    s_query = int(data[index]) - 1  # Convert to 0-based
    index += 1
    t_query = int(data[index]) - 1  # Convert to 0-based
    index += 1
    queries.append((s_query, t_query))

# Precompute data structures for R sort
idx_R_sort = sorted(range(N), key=lambda i: R[i])
W_R = [W[idx] for idx in idx_R_sort]
prefix_min_W_R = [W_R[0]]
for i in range(1, N):
    prefix_min_W_R.append(min(prefix_min_W_R[-1], W_R[i]))
R_sorted_val = [R[idx] for idx in idx_R_sort]  # Sorted R values

# Precompute data structures for L sort
idx_L_sort = sorted(range(N), key=lambda i: L[i])
W_L_sorted = [W[idx] for idx in idx_L_sort]
suffix_min_W_L = [0] * N
suffix_min_W_L[N - 1] = W_L_sorted[N - 1]
for i in range(N - 2, -1, -1):
    suffix_min_W_L[i] = min(W_L_sorted[i], suffix_min_W_L[i + 1])
L_sorted_val = [L[idx] for idx in idx_L_sort]  # Sorted L values

# Process each query
for s_idx, t_idx in queries:
    L_s = L[s_idx]
    R_s = R[s_idx]
    L_t = L[t_idx]
    R_t = R[t_idx]
    
    # Check if intervals are disjoint
    if R_s < L_t or R_t < L_s:  # Disjoint
        dist = W[s_idx] + W[t_idx]
        print(dist)
    else:  # Intersect
        C = min(L_s, L_t)
        D = max(R_s, R_t)
        
        # Min W_u with R_u < C
        bis_left_C = bisect.bisect_left(R_sorted_val, C)
        if bis_left_C > 0:
            min_left_val = prefix_min_W_R[bis_left_C - 1]
        else:
            min_left_val = float('inf')
        
        # Min W_u with L_u > D
        bis_right_D = bisect.bisect_right(L_sorted_val, D)  # First index where L > D
        if bis_right_D < N:
            min_right_val = suffix_min_W_L[bis_right_D]
        else:
            min_right_val = float('inf')
        
        # Min over u
        min_W_u = min(min_left_val, min_right_val)
        
        if min_W_u < float('inf'):  # There is some u
            dist = W[s_idx] + W[t_idx] + min_W_u
            print(dist)
        else:
            print(-1)