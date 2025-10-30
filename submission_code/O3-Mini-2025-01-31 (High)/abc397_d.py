def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    try:
        N = int(data[0])
    except:
        return

    # We use the fact that if x^3 - y^3 = N with x > y > 0,
    # then letting d = x - y we have:
    #   (y + d)^3 - y^3 = d * (3y^2 + 3d*y + d^2) = N.
    # In particular, d divides N.
    #
    # We then set m = N // d and our equation becomes:
    #   3y^2 + 3d*y + d^2 = m.
    # Rearranging:
    #   3y^2 + 3d*y + (d^2 - m) = 0.
    # The quadratic formula for y is then:
    #   y = (-3d ± sqrt(9d^2 - 12*(d^2 - m))) / 6
    #     = (-3d ± sqrt(12*m - 3*d^2)) / 6.
    # We need the positive solution:
    #   y = (-3d + r) / 6,  where r = sqrt(12*m - 3*d^2)
    # and we must have r an integer and (r - 3*d) divisible by 6.
    #
    # Also, note that for a fixed d, the minimum value for N happens when y = 1:
    #   N = d*(3*1^2 + 3*d*1 + d^2) = d*(d^2 + 3d + 3).
    # In particular, d^3 is a lower bound so that d cannot exceed the integer cube root of N.
    #
    # First, compute d_max = floor(cube_root(N)).
    lo, hi = 1, int(10**6) + 10  # for N up to 1e18, cube root is at most 1e6.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * mid * mid <= N:
            lo = mid
        else:
            hi = mid - 1
    d_max = lo

    found = False
    ans_x = -1
    ans_y = -1

    # Try all possible values for d = x - y from 1 up to d_max.
    for d in range(1, d_max + 1):
        if N % d != 0:
            continue
        m = N // d
        # The quadratic in y is: 3y^2 + 3d*y + (d^2 - m) = 0.
        # Discriminant, Δ, is computed as:
        #   Δ = (3d)^2 - 4*3*(d^2 - m) = 9d^2 - 12d^2 + 12m
        #     = 12*m - 3*d^2 = 3 * (4*m - d^2)
        disc = 12 * m - 3 * d * d
        if disc < 0:
            continue
        r = math.isqrt(disc)
        if r * r != disc:
            continue
        # For a valid positive integer solution, we need r > 3*d and (r - 3*d) divisible by 6.
        if r <= 3 * d or (r - 3*d) % 6 != 0:
            continue
        y = (r - 3 * d) // 6
        if y <= 0:
            continue
        x = y + d
        # Double-check the solution:
        if x**3 - y**3 == N:
            ans_x = x
            ans_y = y
            found = True
            break

    if found:
        sys.stdout.write(f"{ans_x} {ans_y}")
    else:
        sys.stdout.write("-1")


if __name__ == '__main__':
    main()