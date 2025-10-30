# YOUR CODE HERE
import sys
import math

def get_prime_factors(x):
    factors = {}
    while x % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        x = x // 2
    i = 3
    while i * i <= x:
        while x % i == 0:
            factors[i] = factors.get(i, 0) + 1
            x = x // i
        i += 2
    if x > 2:
        factors[x] = 1
    return factors

def count_moves(x):
    factors = get_prime_factors(x)
    total = 0
    for p in factors:
        total += factors[p]
    return total

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    xor_sum = 0
    for a in A:
        moves = count_moves(a) - 1
        xor_sum ^= moves
    if xor_sum != 0:
        print("Anna")
    else:
        print("Bruno")

if __name__ == "__main__":
    main()