# Read input
N = int(input())
A = list(map(int, input().split()))

# Calculate the sum of f(A_i, A_j) for all pairs i < j
total_sum = 0
MOD = 10**8

for i in range(N-1):
    for j in range(i+1, N):
        total_sum += (A[i] + A[j]) % MOD

# Print the answer
print(total_sum)