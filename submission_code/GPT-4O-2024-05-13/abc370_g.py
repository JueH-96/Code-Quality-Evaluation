# YOUR CODE HERE
import sys
input = sys.stdin.read
MOD = 998244353

def sum_of_divisors(n):
    total = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total

def is_good_integer(n):
    return sum_of_divisors(n) % 3 == 0

def count_good_integers(N):
    count = 0
    for i in range(1, N + 1):
        if is_good_integer(i):
            count += 1
    return count

def mod_exp(base, exp, mod):
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def main():
    data = input().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    good_count = count_good_integers(N)
    result = mod_exp(good_count, M, MOD)
    
    print(result)

if __name__ == "__main__":
    main()