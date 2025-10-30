import sys
import math

def floor_cuberoot(n: int) -> int:
    """
    Returns the largest integer c such that c**3 <= n < (c+1)**3
    """
    c = int(round(n ** (1 / 3)))          # close to the true root
    while (c + 1) ** 3 <= n:              # adjust up if we rounded down
        c += 1
    while c ** 3 > n:                     # adjust down if we rounded up
        c -= 1
    return c


def main() -> None:
    N = int(sys.stdin.readline().strip())

    #  x - y  (= d) cannot exceed the cubic root of N
    limit = floor_cuberoot(N)

    for d in range(1, limit + 1):
        if N % d:                         # d must divide N
            continue

        K = N // d                        # K = x^2 + x y + y^2

        # Discriminant of 3y^2 + 3dy + d^2 - K = 0
        D = 12 * K - 3 * d * d
        if D < 0:
            continue

        w = math.isqrt(D)                 # integer square root
        if w * w != D:                    # D must be a perfect square
            continue

        if w <= 3 * d:                    # y has to be positive
            continue
        if (w - 3 * d) % 6:               # y must be integer
            continue

        y = (w - 3 * d) // 6
        if y <= 0:
            continue

        x = y + d
        # Final safety check (guards against overflow/logic slips)
        if x ** 3 - y ** 3 == N:
            print(f"{x} {y}")
            return

    print(-1)


if __name__ == "__main__":
    main()