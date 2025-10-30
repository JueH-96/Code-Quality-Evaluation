n = int(input())
A = list(map(int, input().split()))
INF = 1e18

min_left = [INF] * n
min_left[0] = INF  # j=0 has no left elements

for j in range(1, n):
    current = A[j-1] - (j-1)
    min_left[j] = min(min_left[j-1], current)

min_right = [INF] * n
min_right[-1] = INF  # j=n-1 has no right elements

for j in range(n-2, -1, -1):
    current = A[j+1] + (j+1)
    min_right[j] = min(min_right[j+1], current)

max_k = 0

for j in range(n):
    # Compute current_left and current_right
    if min_left[j] != INF:
        current_left = min_left[j] + j
    else:
        current_left = INF
    
    if min_right[j] != INF:
        current_right = min_right[j] - j
    else:
        current_right = INF
    
    # Compute k_candidate
    k_candidate = min(A[j], current_left, current_right)
    
    # Compute window constraints
    window_k_max = min(j + 1, n - j)
    
    # Compute k_max_j
    k_max_j = min(k_candidate, window_k_max)
    
    if k_max_j > max_k:
        max_k = k_max_j

print(max_k)