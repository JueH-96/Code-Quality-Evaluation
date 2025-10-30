def solve(N, M, intervals):
    # Sort intervals by L_i in decreasing order
    intervals.sort(key=lambda x: x[0], reverse=True)
    
    # Initialize min_R array
    min_R = [M + 1] * (M + 1)
    
    # Compute minimum R_i for each L_i
    for L_i, R_i in intervals:
        min_R[L_i] = min(min_R[L_i], R_i)
    
    # Compute running minima
    for l in range(M - 1, 0, -1):
        min_R[l] = min(min_R[l], min_R[l + 1])
    
    count = 0
    for l in range(1, M + 1):
        max_r = min_R[l] - 1
        if max_r >= l:
            count += max_r - l + 1
    
    return count

# Read input
N, M = map(int, input().split())
intervals = []
for _ in range(N):
    L_i, R_i = map(int, input().split())
    intervals.append((L_i, R_i))

print(solve(N, M, intervals))