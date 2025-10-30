import math

def main():
    import sys
    N, M, K = map(int, sys.stdin.readline().split())
    g = math.gcd(N, M)
    lcm = (N // g) * M

    def count(x):
        return (x // N) + (x // M) - 2 * (x // lcm)

    low = 1
    high = 1
    # Expand the upper bound until count(high) >= K
    while count(high) < K:
        high *= 2

    # Binary search to find the smallest x with count(x) >= K
    while low < high:
        mid = (low + high) // 2
        c = count(mid)
        if c < K:
            low = mid + 1
        else:
            high = mid

    print(low)

if __name__ == "__main__":
    main()