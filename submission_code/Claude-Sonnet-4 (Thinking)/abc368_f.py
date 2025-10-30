def prime_factorization_sum(n):
    if n == 1:
        return 0
    
    sum_exp = 0
    i = 2
    while i * i <= n:
        while n % i == 0:
            sum_exp += 1
            n //= i
        i += 1
    if n > 1:
        sum_exp += 1
    return sum_exp

n = int(input())
a = list(map(int, input().split()))

xor_sum = 0
for num in a:
    xor_sum ^= prime_factorization_sum(num)

if xor_sum == 0:
    print("Bruno")
else:
    print("Anna")