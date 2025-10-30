n = int(input())
a = list(map(int, input().split()))

MOD = 998244353

# Calculate number of digits for each element
digits = [len(str(x)) for x in a]

# Calculate the second part: sum of A_j * j for j from 1 to n-1 (0-indexed)
second_part = 0
for j in range(1, n):
    second_part = (second_part + a[j] * j) % MOD

# Calculate suffix sums of powers
suffix_sum = [0] * n
for k in range(n-2, -1, -1):
    suffix_sum[k] = (suffix_sum[k+1] + pow(10, digits[k+1], MOD)) % MOD

# Calculate the first part: sum of A_i * suffix_sum[i]
first_part = 0
for i in range(n-1):
    first_part = (first_part + a[i] * suffix_sum[i]) % MOD

answer = (first_part + second_part) % MOD
print(answer)