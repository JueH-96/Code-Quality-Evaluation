def main():
    import sys, math

    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # All numbers that can be represented as a^b (b>=2) and that are perfect powers include:
    #   - All perfect squares, i.e. numbers of the form a^2. (Note that 1 = 1^2 is included.)
    #   - Other perfect powers that are not squares.
    #
    # We can count all perfect squares directly, because the count is floor(sqrt(N)).
    # A number x = a^b is a square if b is even (since a^(even b) = (a^(b/2))^2).
    # Thus, numbers coming from any even exponent are already counted.
    #
    # For odd exponents (b = 3, 5, 7, ...), if the base a is not a perfect square then a^b is not a square.
    # We loop over odd exponents b >= 3. For each such b, we only loop over bases a from 2 up to the
    # maximum integer such that a^b <= N (we get that using an integer nth root function).
    # We also use a set to avoid any duplicates (for example, a number like 512 = 8^3 = 2^9).
    
    # Count of perfect squares:
    count_squares = math.isqrt(N)
    
    extras = set()  # to store perfect powers not already counted (non-squares)

    # Maximum possible exponent b: x = a^b <= N implies b <= log_2(N) (because for a >= 2,
    # 2^b <= N). We thus set max_b = floor(log_2(N)).
    if N > 1:
        max_b = int(math.log(N, 2))
    else:
        max_b = 2

    # For b = 3, 5, 7, ... up to max_b:
    for b in range(3, max_b + 1):
        if b % 2 == 0:
            continue  # a^b with even exponent is a perfect square and already counted.
        max_a = integer_nth_root(N, b)
        # a = 1 gives 1 which is already counted in squares, so start from 2.
        for a in range(2, max_a + 1):
            # If a itself is a perfect square, then a^b will be a square.
            if is_perfect_square(a):
                continue
            x = pow(a, b)  # Guaranteed x <= N by design of max_a.
            extras.add(x)
    
    ans = count_squares + len(extras)
    sys.stdout.write(str(ans))

def integer_nth_root(n, k):
    # Returns the largest integer a such that a**k <= n using binary search.
    lo = 1
    hi = int(n**(1.0/k)) + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        p = mid**k
        if p == n:
            return mid
        elif p < n:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi

def is_perfect_square(x):
    # Checks if x is a perfect square.
    r = int(math.isqrt(x))
    return r * r == x

if __name__ == '__main__':
    main()