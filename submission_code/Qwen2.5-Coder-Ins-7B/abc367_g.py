import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
A = list(map(int, data[3:]))

MOD = 998244353

# Function to calculate (base^exp) % mod using iterative method
def power(base, exp, mod):
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

# Function to calculate the sum of scores of all non-empty subsequences
def sum_of_scores(N, M, K, A):
    total_sum = 0
    total_subsequences = 1 << N
    for i in range(1, total_subsequences):
        subsequence = []
        for j in range(N):
            if i & (1 << j):
                subsequence.append(A[j])
        subsequence_length = len(subsequence)
        if subsequence_length % M == 0:
            xor_value = 0
            for num in subsequence:
                xor_value ^= num
            total_sum += power(xor_value, K, MOD)
            total_sum %= MOD
    return total_sum

# Calculate and print the result
result = sum_of_scores(N, M, K, A)
print(result)