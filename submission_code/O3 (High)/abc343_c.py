import sys
import math

def is_palindrome(num: int) -> bool:
    s = str(num)
    return s == s[::-1]

def integer_cuberoot(n: int) -> int:
    """
    Returns floor(cuberoot(n)) for non-negative n using binary search.
    """
    lo, hi = 0, 10**6 + 1          # because (10**6)^3 = 10**18
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * mid * mid <= n:
            lo = mid
        else:
            hi = mid
    return lo

def main() -> None:
    n = int(sys.stdin.readline().strip())

    limit = integer_cuberoot(n)     # largest x with x^3 ≤ n
    best  = 0

    for x in range(1, limit + 1):
        cube = x * x * x
        if is_palindrome(cube):
            best = cube             # cubes increase with x, so last match is the largest

    print(best)

if __name__ == "__main__":
    main()