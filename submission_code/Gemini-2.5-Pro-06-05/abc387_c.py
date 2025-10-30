import sys

def solve():
    """
    Reads the input range [L, R], calculates the number of Snake numbers
    within this range (inclusive), and prints the result.
    """
    L, R = map(int, sys.stdin.readline().split())

    def calculate(N_str: str) -> int:
        """
        Counts the number of Snake numbers between 1 and N (inclusive).
        This is a combinatorial digit counting approach.

        Args:
            N_str: The upper bound N as a string.

        Returns:
            The total count of Snake numbers up to N.
        """
        n = len(N_str)
        res = 0

        # Part 1: Count Snake numbers with fewer digits than N.
        # These numbers have length k, where 2 <= k < n.
        for k in range(2, n):
            # The first digit 'd' can be from 1 to 9.
            for d in range(1, 10):
                # The remaining k-1 digits have 'd' choices each (0 to d-1).
                res += pow(d, k - 1)

        # Part 2: Count Snake numbers with the same number of digits as N.
        if n >= 2:
            s = N_str
            d0 = int(s[0])
            
            # 2a: Count numbers where the first digit is smaller than N's first digit.
            for d in range(1, d0):
                # For each such first digit 'd', the remaining n-1 digits have 'd' choices.
                res += pow(d, n - 1)

            # 2b: Count numbers where the prefix matches N's prefix.
            top_d = d0
            is_prefix_valid_snake = True
            for i in range(1, n):
                current_digit_limit = int(s[i])
                
                # Count numbers that are smaller than N because of the digit at position 'i'.
                # The digit must be < current_digit_limit and also < top_d.
                limit = min(current_digit_limit, top_d)
                if limit > 0:
                    # 'limit' choices for the digit at 'i'. Remaining digits have 'top_d' choices.
                    res += limit * pow(top_d, n - 1 - i)
                
                # Check if N's prefix itself can form a Snake number.
                if current_digit_limit >= top_d:
                    is_prefix_valid_snake = False
                    break
            
            # 2c: If N's prefix was valid all the way, N itself is a Snake number.
            if is_prefix_valid_snake:
                res += 1
                
        return res

    # The result is count(<=R) - count(<L), which is count(<=R) - count(<=L-1).
    ans_R = calculate(str(R))
    ans_L_minus_1 = calculate(str(L - 1))
    
    print(ans_R - ans_L_minus_1)

solve()