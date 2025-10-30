def main():
    import sys
    import math

    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # We express x^3 - y^3 = N with x = y + d (with d > 0).
    # Then,
    #   (y+d)^3 - y^3 = 3d*y^2 + 3d^2*y + d^3 = N.
    # Factoring d, we have:
    #   d * (3y^2 + 3d*y + d^2) = N.
    # Hence d must divide N. Let M = N/d, then we need:
    #   3y^2 + 3d*y + d^2 = M.
    # This is a quadratic in y:
    #   3y^2 + 3d*y + (d^2 - M) = 0.
    # The discriminant (for integer y) is:
    #   Δ = (3d)^2 - 4 * 3 * (d^2 - M)
    #      = 9d^2 - 12d^2 + 12M = 12M - 3d^2.
    # For an integer solution to exist, Δ must be nonnegative and a perfect square.
    # We then obtain y = (-3d + sqrt(Δ)) / 6, and must have y > 0.
    #
    # d = x-y must be at least 1 and at most about cube-root of N (since for y>=1,
    # d*(3y^2 + 3d*y + d^2) is roughly at least d^3 when y is small).
    
    # We set an upper bound for d based on the cube root of N:
    d_max = int(round(N ** (1/3))) + 2
    
    for d in range(1, d_max):
        if N % d != 0:
            continue
        M = N // d
        # Compute the discriminant
        disc = 12 * M - 3 * d * d
        if disc < 0:
            continue
        s = math.isqrt(disc)
        if s * s != disc:
            continue
        # Check if y is an integer:
        if (-3*d + s) % 6 != 0:
            continue
        y = (-3*d + s) // 6
        if y <= 0:
            continue
        x = y + d
        if x**3 - y**3 == N:
            sys.stdout.write(f"{x} {y}")
            return
    sys.stdout.write("-1")

if __name__ == '__main__':
    main()