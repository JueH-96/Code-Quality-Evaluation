import sys
import math
from functools import lru_cache

MOD = 998244353

def sum_of_divisors(n):
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sum(divisors)

def is_good_integer(n):
    return sum_of_divisors(n) % 3 == 0

@lru_cache(None)
def count_sequences(N, M):
    if M == 1:
        count = 0
        for i in range(1, N + 1):
            if is_good_integer(i):
                count += 1
        return count

    count = 0
    for i in range(1, N + 1):
        if N // i >= i:
            count += count_sequences(N // i, M - 1)
        else:
            break
    return count % MOD

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])

    result = count_sequences(N, M)
    print(result)

if __name__ == "__main__":
    main()