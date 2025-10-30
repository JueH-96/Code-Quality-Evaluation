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

def count_good_integers_up_to(N):
    count = 0
    for i in range(1, N + 1):
        if is_good_integer(i):
            count += 1
    return count

def modular_exponentiation(base, exp, mod):
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def solve():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, M = map(int, data.split())
    
    MOD = 998244353
    
    # Count how many good integers are there up to N
    good_count = count_good_integers_up_to(N)
    
    # We need to find the number of sequences of length M where the product is a good integer
    # Since the product of any sequence of length M where each element is a good integer is also a good integer,
    # we just need to count all sequences of length M where each element is a good integer.
    # This is simply good_count^M (each element in the sequence can be any of the good integers)
    
    result = modular_exponentiation(good_count, M, MOD)
    print(result)