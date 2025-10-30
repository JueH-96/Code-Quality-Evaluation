def is_feasible(H, U, D, X, N):
    ranges = []
    for i in range(N):
        L = max(0, H - D[i])
        R = min(U[i], H)
        if L > R:
            return False
        ranges.append((L, R))
    
    feasible_range = ranges[0]
    for i in range(1, N):
        L, R = ranges[i]
        new_L = max(L, feasible_range[0] - X)
        new_R = min(R, feasible_range[1] + X)
        if new_L > new_R:
            return False
        feasible_range = (new_L, new_R)
    
    return True

N, X = map(int, input().split())
U = []
D = []
for _ in range(N):
    u, d = map(int, input().split())
    U.append(u)
    D.append(d)

total_sum = sum(U[i] + D[i] for i in range(N))
min_sum = min(U[i] + D[i] for i in range(N))

left, right = 1, min_sum
max_H = 0

while left <= right:
    mid = (left + right) // 2
    if is_feasible(mid, U, D, X, N):
        max_H = mid
        left = mid + 1
    else:
        right = mid - 1

print(total_sum - N * max_H)