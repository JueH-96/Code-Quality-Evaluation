def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    L = int(data[0])
    R = int(data[1])
    
    # F(x) counts the number of Snake numbers (>=10) that are <= x.
    def F(x):
        # Since snake numbers are defined for numbers >=10,
        # if x is less than 10, there are none.
        if x < 10:
            return 0
        s = str(x)
        n = len(s)
        total = 0
        # Count snake numbers having fewer digits than x.
        # For an m-digit snake number (m >= 2), the number is formed by
        # choosing a first digit d (from 1 to 9) and then each of the remaining 
        # m-1 digits from 0 to d-1. Thus for a fixed m, count = sum(d^(m-1) for d in 1..9).
        for m in range(2, n):
            subtotal = 0
            for d in range(1, 10):
                subtotal += d ** (m - 1)
            total += subtotal
        
        # Now count snake numbers with exactly n digits that are <= x.
        # Let the first digit of x be fd. For any candidate snake number of length n,
        # if its first digit is less than fd, then any combination of the remaining digits
        # (from 0 to that first digit minus one) will yield a valid snake number.
        first_digit = int(s[0])
        same_len = 0
        for d in range(1, first_digit):
            same_len += d ** (n - 1)
        
        # For numbers whose first digit equals first_digit, we must count only those
        # completions that do not exceed the suffix of x.
        # For a snake number, after choosing first digit A, each subsequent digit must be in [0, A-1].
        # We now use a digit-DP over the remaining positions (positions 1 .. n-1)
        # to count how many completions are <= the corresponding digits in x.
        A = first_digit  # Allowed digits for positions 1..n-1: 0,1,...,A-1.
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dp(pos, tight):
            # pos: current index in s (with index 0 fixed as the first digit)
            # tight: True if so far our chosen digits equal the corresponding digits of x.
            if pos == n:
                return 1
            if not tight:
                # When not tight, each of the remaining positions can be any digit in [0, A-1]
                return A ** (n - pos)
            cur = int(s[pos])
            # If the current digit of x is greater than the maximum allowed (A-1),
            # then no matter what digit (from 0 to A-1) we choose, it will be less
            # than s[pos] so that the rest positions are free.
            if cur > A - 1:
                return A ** (n - pos)
            # Otherwise, cur is in the allowed range.
            count = 0
            # For any choice of a digit d with 0 <= d < cur, the number at this position is 
            # less than s[pos] so subsequent digits can be chosen arbitrarily.
            count += cur * (A ** (n - pos - 1))
            # And if we choose d exactly equal to cur, we must continue the tight condition.
            count += dp(pos + 1, True)
            return count
        
        same_len += dp(1, True)
        total += same_len
        return total

    # The number of snake numbers in [L, R] equals F(R) - F(L-1).
    ans = F(R) - F(L - 1)
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()