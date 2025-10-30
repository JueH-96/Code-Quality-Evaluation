n = int(input())
s = input().strip()

MOD = 10**9 + 7  # Not used in this approach, but can be considered for other optimizations

# Precompute powers of 10 up to 10^(n)
pow10 = [1] * (n + 2)
for i in range(1, n + 2):
    pow10[i] = pow10[i-1] * 10

total = 0
for m in range(n):
    digit = int(s[m])
    exponent = n - m
    sum_10 = (pow10[exponent] - 1) // 9
    contribution = digit * (m + 1) * sum_10
    total += contribution

print(total)