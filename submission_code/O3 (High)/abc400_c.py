import sys
import math

def main() -> None:
    N = int(sys.stdin.readline().strip())

    ans = 0
    power_of_two = 2          # 2^1   (a must be positive)

    # 2^60  ≈ 1.15×10^18 > 10^18  ⇒ at most 60 iterations are needed,
    # but we simply loop while 2^a ≤ N.
    while power_of_two <= N:
        # Maximum value an odd b can take
        #     power_of_two * b^2 ≤ N  ⇒  b ≤ √(N / power_of_two)
        limit = N // power_of_two
        root  = math.isqrt(limit)          # floor(√limit)

        # number of odd integers in [1 … root] is (root + 1) // 2
        ans += (root + 1) // 2

        power_of_two <<= 1                 # next power of two (multiply by 2)

    print(ans)

if __name__ == "__main__":
    main()