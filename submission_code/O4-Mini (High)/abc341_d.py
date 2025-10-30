import sys
import threading

def main():
    import math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    # gcd and lcm
    g = math.gcd(N, M)
    lcm = N // g * M

    # count(X): number of positive integers <= X divisible by exactly one of N or M
    def count(x):
        return x // N + x // M - 2 * (x // lcm)

    # Binary search for the smallest X such that count(X) >= K
    lo = 0
    hi = K * max(N, M)
    # Invariant: count(lo) < K, count(hi) >= K
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) >= K:
            hi = mid
        else:
            lo = mid + 1

    # lo is our answer
    sys.stdout.write(str(lo))

if __name__ == "__main__":
    main()