import sys

def max_b(M):
    left = 0
    right = M
    while left <= right:
        mid = (left + right) // 2
        if mid * (mid + 1) <= M:
            left = mid + 1
        else:
            right = mid - 1
    return right

data = sys.stdin.read().strip()
R = int(data)
total = 0
K = R * R - 1
for a in range(0, R):
    M_a = K - a * (a + 1)
    max_b_a = max_b(M_a)
    sum_b_num_j = 1 + 2 * max_b_a
    num_i_a = 1 if a == 0 else 2
    contrib = num_i_a * sum_b_num_j
    total += contrib
print(total)