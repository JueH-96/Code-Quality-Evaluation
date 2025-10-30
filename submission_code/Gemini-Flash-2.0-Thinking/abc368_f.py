def count_prime_factors(n):
    count = 0
    d = 2
    temp_n = n
    while d * d <= temp_n:
        while temp_n % d == 0:
            count += 1
            temp_n //= d
        d += 1
    if temp_n > 1:
        count += 1
    return count

n = int(input())
a = list(map(int, input().split()))

xor_sum = 0
for x in a:
    xor_sum ^= count_prime_factors(x)

if xor_sum > 0:
    print("Anna")
else:
    print("Bruno")