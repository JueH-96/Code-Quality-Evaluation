import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])

    # Compute gcd and lcm
    from math import gcd
    g = gcd(N, M)
    lcm = N // g * M

    # Function to count numbers <= x divisible by exactly one of N and M
    def count_exact(x):
        return x // N + x // M - 2 * (x // lcm)

    # Binary search for the smallest x such that count_exact(x) >= K
    lo = 1
    hi = K * max(N, M)  # upper bound for the K-th number
    while lo < hi:
        mid = (lo + hi) // 2
        if count_exact(mid) >= K:
            hi = mid
        else:
            lo = mid + 1

    # lo is the K-th number divisible by exactly one of N and M
    print(lo)

if __name__ == "__main__":
    main()