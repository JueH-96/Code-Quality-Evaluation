import math

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    K = int(input[2])

    # Compute GCD of N and M
    gcd = math.gcd(N, M)
    # Compute LCM
    lcm = (N * M) // gcd

    def count(x):
        a = x // N
        b = x // M
        c = x // lcm
        return a + b - 2 * c

    low = 1
    high = 10**18  # A large enough upper bound

    while low < high:
        mid = (low + high) // 2
        current = count(mid)
        if current < K:
            low = mid + 1
        else:
            high = mid

    print(low)

if __name__ == '__main__':
    main()