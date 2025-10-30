import sys

data = sys.stdin.read().split()
N = int(data[0])
S = data[1]

a = int('1' * N)  # Represents (10^N - 1) // 9
total_sum = 0

for i in range(N):
    digit_val = int(S[i])
    k_val = i + 1
    contrib = (digit_val * k_val) * a
    total_sum += contrib
    a = (a - 1) // 10

print(total_sum)