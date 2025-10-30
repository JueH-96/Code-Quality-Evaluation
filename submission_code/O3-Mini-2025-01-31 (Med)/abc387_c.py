def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    L = int(data[0])
    R = int(data[1])
    
    # A Snake number is a positive integer (at least two digits)
    # whose first (most significant) digit a is strictly larger than every other digit.
    # For a fixed number of digits d (with d >= 2) and with top digit a (1 ≤ a ≤ 9),
    # every other digit can be any number in 0..(a-1).
    # So the count of d-digit snake numbers with top digit a is exactly a^(d-1)
    # and the total count (for exactly d digits) is:
    #      f(d) = sum_{a=1}^{9} a^(d-1)
    #
    # To count the snake numbers in an interval [L, R] (with R up to 10^18),
    # we count how many snake numbers are ≤ x (using a function count_snake(x))
    # and then subtract: answer = count_snake(R) - count_snake(L-1).
    #
    # For any x, if x has n digits then for every snake number with fewer than n digits,
    # they are automatically ≤ x. For numbers with exactly n digits we must count
    # only those that are <= x. We do that with a kind of digit‐DP that “fixes”
    # the first digit (which becomes our top digit) and then forces every later
    # digit to lie in 0..(top-1).
    
    # dp(pos, tight, top, digits, n)
    #    pos: current index (starting at 1 because pos=0, the most significant digit, is handled separately)
    #    tight: True if the prefix we built equals the prefix of x so far.
    #    top: the chosen top digit (from the first digit, which is never 0)
    #    digits: list of digits of x (most-significant first)
    #    n: total length of x
    #
    # When not in the “tight” region, every remaining digit can be any of 0..(top-1);
    # thus the number of completions is top^(remaining_positions).
    
    def dp(pos, tight, top, digits, n):
        if pos == n:
            return 1
        if not tight:
            return top ** (n - pos)
        res = 0
        limit = digits[pos]
        max_allowed = top - 1  # every digit must be strictly less than the top digit
        # Allowed digit for this position is in 0 .. min(max_allowed, limit).
        upp = min(max_allowed, limit)
        # For any choice d < limit (and d in [0, upp-1]), the remainder can be filled arbitrarily.
        for d in range(0, upp):
            res += top ** (n - pos - 1)
        # If the digit equal to the limit is allowed (i.e. limit <= max_allowed), we continue in tight mode.
        if limit <= max_allowed:
            res += dp(pos + 1, True, top, digits, n)
        return res

    def fixed_count(x_str):
        # Count how many snake numbers with exactly len(x_str) digits are ≤ int(x_str)
        digits = list(map(int, x_str))
        n = len(digits)
        if n < 2:
            return 0
        res = 0
        # For the first digit (the most significant digit):
        # We must choose a digit between 1 and 9.
        # If we choose a number less than the first digit of x,
        # then the rest of the (n-1) digits can be chosen arbitrarily.
        first_lim = digits[0]
        for d in range(1, first_lim):
            res += d ** (n - 1)
        # For the candidate equal to the first digit, we continue with the DP.
        res += dp(1, True, digits[0], digits, n)
        return res

    def full_count_for_digits(d):
        # The total number of snake numbers with exactly d digits:
        total = 0
        for a in range(1, 10):
            total += a ** (d - 1)
        return total

    def count_snake(x):
        # Count the snake numbers that are ≤ x.
        # Note: by definition, snake numbers are at least two digits, so for x < 10, the answer is zero.
        if x < 10:
            return 0
        s = 0
        x_str = str(x)
        n = len(x_str)
        # For numbers with fewer than n digits, all snake numbers qualify.
        for d in range(2, n):
            s += full_count_for_digits(d)
        # For numbers with exactly n digits, use our DP.
        s += fixed_count(x_str)
        return s

    result = count_snake(R) - count_snake(L - 1)
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()