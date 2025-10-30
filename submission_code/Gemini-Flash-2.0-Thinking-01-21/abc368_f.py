import sys

def get_prime_factor_exponent_sum(n):
    exponent_sum = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            exponent_sum += 1
            n //= d
        d += 1
    if n > 1:
        exponent_sum += 1
    return exponent_sum

def solve():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    grundy_values = []
    for val in a:
        grundy_values.append(get_prime_factor_exponent_sum(val))
    
    xor_sum = 0
    for g_val in grundy_values:
        xor_sum ^= g_val
        
    if xor_sum == 0:
        print("Bruno")
    else:
        print("Anna")

if __name__ == '__main__':
    solve()