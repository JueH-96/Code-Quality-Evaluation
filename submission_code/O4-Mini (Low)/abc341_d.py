import sys
import threading

def main():
    import sys
    import math

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])

    # Compute gcd and lcm
    g = math.gcd(N, M)
    l = N // g * M

    # Count of numbers <= x divisible by exactly one of N, M
    def count(x):
        # divisible by N only: x//N - x//l
        # divisible by M only: x//M - x//l
        return (x // N + x // M - 2 * (x // l))

    # Binary search smallest x with count(x) >= K
    lo = 1
    hi = K * min(N, M)  # safe upper bound
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < K:
            lo = mid + 1
        else:
            hi = mid
    print(lo)

if __name__ == "__main__":
    main()