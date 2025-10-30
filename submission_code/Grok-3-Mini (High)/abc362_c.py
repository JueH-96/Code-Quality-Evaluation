import sys
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1
L = [0] * N
R = [0] * N
for i in range(N):
    L[i] = data[index]
    index += 1
    R[i] = data[index]
    index += 1

sum_L = sum(L)
sum_R = sum(R)

if not (sum_L <= 0 <= sum_R):
    print("No")
else:
    X = list(L)  # Copy of L
    delta_needed = -sum_L  # Amount to increase the sum
    for i in range(N):
        if delta_needed > 0:
            max_can_add = R[i] - X[i]
            add_amount = min(max_can_add, delta_needed)
            X[i] += add_amount
            delta_needed -= add_amount
    # Now output the result
    print("Yes")
    print(*X)