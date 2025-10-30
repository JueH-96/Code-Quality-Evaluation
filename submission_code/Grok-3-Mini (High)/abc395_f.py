import sys
sys.setrecursionlimit(1000000)

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
X = int(data[index])
index += 1
U = [0] * N
D = [0] * N
for i in range(N):
    U[i] = int(data[index])
    index += 1
    D[i] = int(data[index])
    index += 1

# Compute min_S and sum_S
min_S = min(U[i] + D[i] for i in range(N))
sum_S = sum(U[i] + D[i] for i in range(N))

def is_feasible(H):
    # Compute A and B lists
    A_list = [max(0, H - D[i]) for i in range(N)]
    B_list = [min(U[i], H) for i in range(N)]
    
    # Compute left lower bound
    left_lb = [0] * N
    left_lb[0] = A_list[0]
    for i in range(1, N):
        left_lb[i] = max(A_list[i], left_lb[i-1] - X)
    
    # Compute right lower bound
    right_lb = [0] * N
    right_lb[N-1] = A_list[N-1]
    for i in range(N-2, -1, -1):
        right_lb[i] = max(A_list[i], right_lb[i+1] - X)
    
    # Compute left upper bound
    left_ub = [0] * N
    left_ub[0] = B_list[0]
    for i in range(1, N):
        left_ub[i] = min(B_list[i], left_ub[i-1] + X)
    
    # Compute right upper bound
    right_ub = [0] * N
    right_ub[N-1] = B_list[N-1]
    for i in range(N-2, -1, -1):
        right_ub[i] = min(B_list[i], right_ub[i+1] + X)
    
    # Check for each i if LB <= UB
    for i in range(N):
        LB_i = max(left_lb[i], right_lb[i])
        UB_i = min(left_ub[i], right_ub[i])
        if LB_i > UB_i:
            return False
    return True

# Binary search for maximum H
low = 0
high = min_S
while low <= high:
    mid = (low + high) // 2
    if is_feasible(mid):
        low = mid + 1
    else:
        high = mid - 1

# high is the maximum H where is_feasible is True
H_max = high
cost = sum_S - N * H_max
print(cost)