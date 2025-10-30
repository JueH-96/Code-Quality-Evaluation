import sys
import math

def main():
    import sys

    N, M, K = map(int, sys.stdin.read().split())

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    lcm = N * M // gcd(N, M)

    def count(x):
        return x // N + x // M - 2 * (x // lcm)

    low = 1
    high = 2 * 10**18
    while low < high:
        mid = (low + high) // 2
        if count(mid) >= K:
            high = mid
        else:
            low = mid + 1
    print(low)

if __name__ == "__main__":
    main()