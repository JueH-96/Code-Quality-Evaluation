def main():
    import sys
    from math import gcd

    data = sys.stdin.read().split()
    if not data:
        return
    N, M, K = map(int, data)

    def lcm(a, b):
        return a * b // gcd(a, b)

    LCM = lcm(N, M)

    # Count the number of positive integers <= x that are divisible by exactly one of N and M.
    def count(x):
        return x // N + x // M - 2 * (x // LCM)

    # Define a safe upper bound. In worst-case, the answer will be at most K * max(N, M).
    low, high = 1, K * max(N, M)

    # Binary search for the smallest x such that count(x) >= K.
    while low < high:
        mid = (low + high) // 2
        if count(mid) >= K:
            high = mid
        else:
            low = mid + 1

    sys.stdout.write(str(low))

if __name__ == '__main__':
    main()