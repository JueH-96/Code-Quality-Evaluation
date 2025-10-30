class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_len = len(s)
        s_num = int(s)
        
        # Function to check if a number is powerful
        def is_powerful(x):
            x_str = str(x)
            if not x_str.endswith(s):
                return False
            for digit in x_str:
                if int(digit) > limit:
                    return False
            return True
        
        # Function to count powerful numbers up to a certain number
        def count_powerful_up_to(n):
            count = 0
            # Generate all possible prefixes
            # The prefix can be any number with digits <= limit, and the length is such that when combined with s, it forms a number <= n
            # The length of the prefix is len(str(n)) - s_len
            n_len = len(str(n))
            prefix_len = n_len - s_len
            if prefix_len < 0:
                return 0
            # Generate all possible prefixes of length prefix_len
            # Each digit of the prefix must be <= limit
            # We can represent the prefix as a number with digits <= limit
            # The number of such prefixes is (limit + 1) ** prefix_len
            # But we need to ensure that the combined number (prefix * 10^s_len + s_num) <= n
            # So we need to find the maximum prefix such that prefix * 10^s_len + s_num <= n
            max_prefix = (n - s_num) // (10 ** s_len)
            # Now, count the number of prefixes <= max_prefix where all digits are <= limit
            # We can treat the prefix as a number with digits <= limit
            # So, the number of such prefixes is the number of numbers with prefix_len digits, each digit <= limit, and <= max_prefix
            # To count this, we can generate all possible numbers with prefix_len digits, each digit <= limit, and count those <= max_prefix
            # But for large prefix_len, this is not efficient
            # Instead, we can use a recursive approach or dynamic programming to count the numbers
            # Here, we use a recursive approach with memoization
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def count_numbers_with_digits(length, max_num):
                if length == 0:
                    return 1
                count = 0
                first_digit = min(limit, max_num // (10 ** (length - 1)))
                for d in range(0, first_digit + 1):
                    count += count_numbers_with_digits(length - 1, max_num - d * (10 ** (length - 1)))
                return count
            return count_numbers_with_digits(prefix_len, max_prefix)
        
        # Calculate the count for finish and start-1, then subtract
        total = count_powerful_up_to(finish)
        if start > 1:
            total -= count_powerful_up_to(start - 1)
        return total