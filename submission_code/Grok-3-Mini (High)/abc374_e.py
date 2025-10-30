import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
X = int(data[index])
index += 1

# Store the processes
A_list = []
P_list = []
B_list = []
Q_list = []
for _ in range(N):
    A = int(data[index])
    index += 1
    P = int(data[index])
    index += 1
    B = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    A_list.append(A)
    P_list.append(P)
    B_list.append(B)
    Q_list.append(Q)

# Define check function
def check(C):
    total_cost = 0
    for i in range(N):
        A_i = A_list[i]
        P_i = P_list[i]
        B_i = B_list[i]
        Q_i = Q_list[i]
        # Compute min cost for this process
        cost_s = ((C + A_i - 1) // A_i) * P_i
        cost_t = ((C + B_i - 1) // B_i) * Q_i
        max_ab = max(A_i, B_i)
        min_mix = float('inf')
        for k in range(0, max_ab + 1):  # 0 to max_ab inclusive
            prod_s = k * A_i
            if prod_s >= C:
                cost = k * P_i
            else:
                remaining = C - prod_s
                m = (remaining + B_i - 1) // B_i
                cost = k * P_i + m * Q_i
            if cost < min_mix:
                min_mix = cost
        min_cost_i = min(cost_s, cost_t, min_mix)
        total_cost += min_cost_i
        if total_cost > X:  # Early termination
            return False
    return total_cost <= X

# Binary search for the maximum C
low = 0
high = 1000000000
result = 0
while low <= high:
    mid = (low + high) // 2
    if check(mid):
        result = mid
        low = mid + 1
    else:
        high = mid - 1

# Output the result
print(result)