import sys
import math

def main() -> None:
    N = int(sys.stdin.readline())

    # 1) count of perfect squares (a^2)
    squares_cnt = math.isqrt(N)          # includes 1 if N >= 1

    # 2) enumerate perfect powers with exponent >= 3
    high_powers = set()
    MAX_EXP = 60                         # 2^60  > 10^18

    for exp in range(3, MAX_EXP + 1):
        # rough upper bound for the base
        base_max = int(N ** (1.0 / exp))

        # correct the possible rounding errors
        while pow(base_max + 1, exp) <= N:
            base_max += 1
        while pow(base_max,     exp) >  N:
            base_max -= 1

        # if even base 2 does not work any more, nothing left for larger exponents
        if base_max < 2:
            break

        for base in range(2, base_max + 1):
            high_powers.add(pow(base, exp))

    # 3) how many of the numbers in `high_powers` are also perfect squares?
    duplicate_cnt = 0
    for v in high_powers:
        r = math.isqrt(v)
        if r * r == v:                   # v is a square, hence already counted in squares_cnt
            duplicate_cnt += 1

    # 4) total distinct perfect powers
    answer = squares_cnt + len(high_powers) - duplicate_cnt
    print(answer)

if __name__ == "__main__":
    main()