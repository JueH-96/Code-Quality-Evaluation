import sys
import math
from collections import defaultdict

def factorize(n):
    factors = {}
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n = n // 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n = n // i
        i += 2
    if n > 2:
        factors[n] = 1
    return factors

def get_square_free(n):
    if n == 0:
        return 0
    factors = factorize(n)
    square_free = 1
    for p, exp in factors.items():
        if exp % 2 != 0:
            square_free *= p
    return square_free

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    count = defaultdict(int)
    for num in A:
        sf = get_square_free(num)
        count[sf] += 1
    total = 0
    for key in count:
        if key == 0:
            c = count[key]
            total += c * (c - 1) // 2
            total += c * (N - c)
        else:
            c = count[key]
            total += c * (c - 1) // 2
    print(total)

if __name__ == "__main__":
    main()