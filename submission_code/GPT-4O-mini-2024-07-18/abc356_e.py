def calculate_sum(N, A):
    total_sum = 0
    for i in range(N):
        for j in range(i + 1, N):
            max_val = max(A[i], A[j])
            min_val = min(A[i], A[j])
            total_sum += max_val // min_val
    return total_sum

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N + 1]))

result = calculate_sum(N, A)
print(result)