MOD = 998244353

def sum_of_divisors(n):
    total = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
        i += 1
    return total

def is_good_integer(n):
    return sum_of_divisors(n) % 3 == 0

def count_good_sequences(N, M):
    good_integers = [i for i in range(1, N + 1) if is_good_integer(i)]
    good_integers_count = len(good_integers)

    if M == 1:
        return good_integers_count % MOD

    dp = [0] * (M + 1)
    dp[0] = 1

    for _ in range(M):
        new_dp = [0] * (M + 1)
        for j in range(M):
            for k in good_integers:
                if j + 1 <= M:
                    new_dp[j + 1] = (new_dp[j + 1] + dp[j]) % MOD
        dp = new_dp

    return dp[M]

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Calculate and print the result
result = count_good_sequences(N, M)
print(result)