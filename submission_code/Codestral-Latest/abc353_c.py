# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

MOD = 10**8

# Calculate the sum of f(A_i, A_j) for all i < j
total_sum = 0
for i in range(N):
    for j in range(i + 1, N):
        total_sum += (A[i] + A[j]) % MOD

print(total_sum)