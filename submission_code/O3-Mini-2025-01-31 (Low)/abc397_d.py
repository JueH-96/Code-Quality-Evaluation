def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # We start by writing x^3 - y^3 = N as:
    #   (x-y)(x^2 + xy + y^2) = N.
    # Let d = x-y > 0, then x = y + d. Substituting, we get:
    #   d * [(y+d)^2 + y(y+d) + y^2] = d * (3y^2 + 3d*y + d^2) = N.
    # Dividing both sides by d (d must divide N), we have:
    #   3y^2 + 3d*y + d^2 = S where S = N/d.
    # This is a quadratic in y:
    #   3y^2 + 3d*y + (d^2 - S) = 0.
    # For integer y to exist, the discriminant must be a perfect square.
    # The discriminant D is:
    #   D = (3d)^2 - 4*3*(d^2 - S) = 9d^2 - 12d^2 + 12S = 12S - 3d^2.
    # We need D >= 0 and a perfect square.
    # Then, y = (-3d + sqrt(D)) / (6)
    # We require y > 0 and integer.
    #
    # The condition D >= 0 gives:
    #   12S - 3d^2 >= 0  =>  4S >= d^2  =>  4N/d >= d^2  => d^3 <= 4N.
    # So, it suffices to iterate d from 1 to cube_root(4N), which is efficient.
    
    max_d = int((4 * N) ** (1/3)) + 2  # small margin
    
    for d in range(1, max_d + 1):
        if N % d != 0:
            continue
        S = N // d
        disc = 12 * S - 3 * d * d
        if disc < 0:
            continue
        # Use math.isqrt to get an integer square root
        t = math.isqrt(disc)
        if t * t != disc:
            continue
        # Now solve for y:
        numerator = -3 * d + t
        if numerator % 6 != 0:
            continue
        y = numerator // 6
        if y <= 0:
            continue
        x = y + d
        # Check correctness:
        if x**3 - y**3 == N:
            sys.stdout.write(f"{x} {y}")
            return
    sys.stdout.write("-1")

if __name__ == "__main__":
    main()