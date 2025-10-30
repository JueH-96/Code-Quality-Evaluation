# YOUR CODE HERE
MOD = 998244353

n = int(input())
a = list(map(int, input().split()))

# Function to count digits
def count_digits(x):
    if x == 0:
        return 1
    count = 0
    while x > 0:
        count += 1
        x //= 10
    return count

# Precompute powers of 10 for each number
powers = []
for num in a:
    digits = count_digits(num)
    powers.append(pow(10, digits, MOD))

result = 0

# For each pair (i, j) where i < j
for i in range(n):
    for j in range(i + 1, n):
        # f(A_i, A_j) = A_i * 10^(digits in A_j) + A_j
        result = (result + a[i] * powers[j] + a[j]) % MOD

print(result)