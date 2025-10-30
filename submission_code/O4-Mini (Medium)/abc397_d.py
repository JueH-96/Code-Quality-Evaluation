import sys
from math import isqrt

def main():
    data = sys.stdin.read().strip()
    if not data:
        print(-1)
        return
    N = int(data)

    # Compute integer cube root r = max a such that a^3 <= N
    # Start from the float approximation
    r = int(N ** (1/3))
    # Adjust if off by one due to float imprecision
    while (r + 1) ** 3 <= N:
        r += 1
    while r ** 3 > N:
        r -= 1

    # Try all positive a = x - y up to r
    for a in range(1, r + 1):
        if N % a != 0:
            continue
        b = N // a
        # We need 3*y^2 + 3*a*y + a^2 = b
        # => 3y^2 + 3a y + (a^2 - b) = 0
        # Discriminant D = 9a^2 - 12*(a^2 - b) = 12b - 3a^2
        D = 12 * b - 3 * a * a
        if D < 0:
            continue
        s = isqrt(D)
        if s * s != D:
            continue
        # y = (-3a + s) / 6, must be integer and > 0
        num = -3 * a + s
        if num <= 0 or num % 6 != 0:
            continue
        y = num // 6
        if y <= 0:
            continue
        x = y + a
        if x <= 0:
            continue
        # We found a valid solution
        print(x, y)
        return

    # If no solution found
    print(-1)

# Call main to execute
main()