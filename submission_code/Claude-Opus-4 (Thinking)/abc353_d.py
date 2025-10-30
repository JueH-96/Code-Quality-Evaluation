# YOUR CODE HERE
def count_digits(n):
    if n == 0:
        return 1
    count = 0
    while n > 0:
        count += 1
        n //= 10
    return count

MOD = 998244353

N = int(input())
A = list(map(int, input().split()))

# Precompute the number of digits for each element
digits = [count_digits(a) for a in A]

# Precompute powers of 10
powers_of_10 = [pow(10, d, MOD) for d in digits]

# Precompute suffix sums of powers of 10
suffix_sum_powers = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    suffix_sum_powers[i] = (suffix_sum_powers[i + 1] + powers_of_10[i]) % MOD

result = 0

# First part: sum_{i=0}^{N-2} A_i * sum_{j=i+1}^{N-1} 10^(digits(A_j))
for i in range(N - 1):
    result = (result + A[i] * suffix_sum_powers[i + 1]) % MOD

# Second part: sum_{j=1}^{N-1} A_j * j
for j in range(1, N):
    result = (result + A[j] * j) % MOD

print(result)