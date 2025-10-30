# YOUR CODE HERE
MOD = 998244353

def count_digits(n):
    return len(str(n))

def power_of_10(exp):
    result = 1
    base = 10
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % MOD
        base = (base * base) % MOD
        exp //= 2
    return result

n = int(input())
A = list(map(int, input().split()))

total = 0

for i in range(n - 1):
    for j in range(i + 1, n):
        # f(A[i], A[j]) = A[i] * 10^(digits in A[j]) + A[j]
        digits_j = count_digits(A[j])
        power_10 = power_of_10(digits_j)
        f_val = (A[i] * power_10 + A[j]) % MOD
        total = (total + f_val) % MOD

print(total)