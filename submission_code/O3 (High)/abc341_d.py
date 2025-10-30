import sys
import math

def main() -> None:
    N, M, K = map(int, sys.stdin.readline().split())

    # Least common multiple of N and M
    g = math.gcd(N, M)
    lcm = N // g * M

    # How many positive integers ≤ x are divisible by exactly one of N or M
    def count(x: int) -> int:
        return x // N + x // M - 2 * (x // lcm)

    # An upper bound that is certainly large enough
    high = K * max(N, M)
    while count(high) < K:          # (rarely needed but keeps us absolutely safe)
        high *= 2

    low = 1
    while low < high:
        mid = (low + high) // 2
        if count(mid) >= K:
            high = mid
        else:
            low = mid + 1

    print(low)

if __name__ == "__main__":
    main()