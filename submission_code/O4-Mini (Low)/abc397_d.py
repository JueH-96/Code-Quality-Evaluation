import sys

def integer_cube_root(n: int) -> int:
    """
    Returns the floor of the cube root of n, i.e. the largest integer x such that x^3 <= n.
    """
    # Initial guess using floating point
    if n < 0:
        # For completeness, though we only call with n >= 0
        x = int(round(-(-n) ** (1/3)))
        x = -x
    else:
        x = int(n ** (1/3))
    # Correct downward if too large
    while x**3 > n:
        x -= 1
    # Correct upward if too small
    while (x + 1)**3 <= n:
        x += 1
    return x

def main():
    data = sys.stdin.read().strip()
    N = int(data)

    # We look for positive integers y up to floor(N^(1/3))
    max_y = int(N ** (1/3)) + 1

    for y in range(1, max_y + 1):
        # We need x^3 = N + y^3
        total = N + y**3
        x = integer_cube_root(total)
        if x > y and x**3 == total:
            # Found a valid solution
            print(x, y)
            return

    # If no solution found, print -1
    print(-1)

if __name__ == "__main__":
    main()