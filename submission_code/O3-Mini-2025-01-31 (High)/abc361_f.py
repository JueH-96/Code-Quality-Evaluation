def integer_nth_root(n, b):
    # Binary search to find the greatest integer a such that a**b <= n.
    lo, hi = 1, n
    while lo <= hi:
        mid = (lo + hi) // 2
        power = mid ** b
        if power == n:
            return mid
        elif power < n:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi

def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # Count the perfect squares up to N.
    # Every number of the form a^2 (with a >= 1) is a perfect power.
    count_squares = math.isqrt(N)
    # We will now count the perfect powers that occur with exponent b >= 3
    # making sure not to re‐count any numbers that are already perfect squares.
    #
    # Observation: If b is even then a^b = (a^(b/2))^2 is a perfect square.
    # Also, if a itself is a perfect square then for any exponent b, a^b is a square.
    # Hence we only consider odd exponents b >= 3 and we skip those bases a that are perfect squares.
    others = set()
    if N < 2:
        # For N = 1, the only perfect power is 1 (since 1 = 1^b for any b ≥ 2).
        sys.stdout.write(str(count_squares))
        return

    # Determine a safe maximum exponent.
    # Since 2^b <= N and N can be as high as 10^18,
    # we have b <= log2(N). For N = 10^18, log2(N) is about 60.
    max_exp = int(math.log2(N))
    
    # Loop over odd exponents b starting from 3.
    for b in range(3, max_exp + 1):
        if b % 2 == 0:
            continue  # Even exponents yield perfect squares already counted.
        a_max = integer_nth_root(N, b)
        if a_max < 2:
            break  # No base a >= 2 will satisfy a**b <= N.
        # Precompute the perfect squares among the possible bases.
        # If a is a perfect square, then a^b becomes a perfect square.
        rmax = int(math.isqrt(a_max))
        squares_for_a = {i * i for i in range(1, rmax + 1)}
        for a in range(2, a_max + 1):
            if a in squares_for_a:
                continue
            # Here x = a^b is not a perfect square.
            x = a ** b
            others.add(x)
    
    # The union of the perfect squares (counted by count_squares)
    # and the other non‐square perfect powers (stored in others) gives the answer.
    ans = count_squares + len(others)
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()