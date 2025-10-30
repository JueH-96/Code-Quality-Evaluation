def mod_exp(base, exp, mod):
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def calculate_score(N, M, K, A):
    MOD = 998244353
    total_score = 0

    # Iterate over all possible non-empty subsequences
    for mask in range(1, 1 << N):
        xor_sum = 0
        count = 0
        
        for i in range(N):
            if mask & (1 << i):
                xor_sum ^= A[i]
                count += 1
        
        if count % M == 0:
            score = mod_exp(xor_sum, K, MOD)
            total_score = (total_score + score) % MOD

    return total_score

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
A = list(map(int, data[3:]))

# Calculate and print the result
result = calculate_score(N, M, K, A)
print(result)