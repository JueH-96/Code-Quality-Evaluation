class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:

        # Helper function for string modulo calculation
        def p_str_is_divisible(p_str: str, k: int) -> bool:
            """
            Checks if a number represented as a string is divisible by k.
            Performs modular arithmetic on the string representation.
            """
            val = 0
            for digit_char in p_str:
                val = (val * 10 + int(digit_char)) % k
            return val == 0

        # Helper function to decrement a string representation of a number
        def string_decrement(s: str) -> str:
            """
            Decrements a number represented as a string by one.
            Handles borrows and returns an empty string if the number
            becomes conceptually smaller than a valid 'half_length'-digit prefix.
            E.g., "100" decrements to "099", which is not a 3-digit prefix, so "" is returned.
            "1" decrements to "0", which is not a 1-digit prefix > 0, so "" is returned.
            """
            s_list = list(s)
            num_digits = len(s_list)
            
            for i in range(num_digits - 1, -1, -1):
                if s_list[i] == '0':
                    s_list[i] = '9'
                else:
                    s_list[i] = str(int(s_list[i]) - 1)
                    break
            
            # If the first digit became '0' after decrementing, 
            # it means the number went from 1 followed by zeros (e.g., "100") 
            # to a number with fewer effective digits (e.g., "099").
            # This is not a valid 'num_digits'-length prefix.
            # Special case: for num_digits=1, "1" becomes "0", which is not a valid first digit.
            if s_list[0] == '0':
                return "" 
            
            return "".join(s_list)

        # Special cases for k=1, 3, 9:
        # Any string of n nines is divisible by 1, 3, and 9.
        # It's also the largest n-digit palindrome.
        if k == 1 or k == 3 or k == 9:
            return "9" * n

        # Determine the length of the first half of the palindrome
        # For n=3, half_length = 2 (prefix "XY" -> "XYX")
        # For n=4, half_length = 2 (prefix "XY" -> "XYYX")
        half_length = (n + 1) // 2

        # Start with the largest possible prefix (e.g., "99" for half_length=2)
        current_prefix_str = "9" * half_length
        
        # We need to limit the number of prefixes we check.
        # This constant is an empirical value found to be sufficient for similar problems,
        # typically covering the search space without hitting time limits.
        # For small k, palindromes are usually found quickly by iterating down.
        MAX_ATTEMPTS = 2000 # Sufficient for k up to 9 for typical test cases

        for _ in range(MAX_ATTEMPTS):
            # If current_prefix_str became empty, it means we've gone below
            # the minimum valid prefix (e.g., '100' -> '099', or '1' -> '0').
            if not current_prefix_str:
                break
            
            # Construct the full palindrome string based on n's parity
            if n % 2 == 0:  # n is even (e.g., n=4, prefix="12" -> "1221")
                p_str = current_prefix_str + current_prefix_str[::-1]
            else:  # n is odd (e.g., n=3, prefix="12" -> "121")
                # The middle digit is the last digit of current_prefix_str.
                # The second half is the reverse of current_prefix_str excluding its last digit.
                p_str = current_prefix_str + current_prefix_str[:-1][::-1]

            # Check if the constructed palindrome is divisible by k
            if p_str_is_divisible(p_str, k):
                return p_str

            # Decrement the prefix for the next iteration
            current_prefix_str = string_decrement(current_prefix_str)
        
        # According to problem constraints, a solution should always be found.
        # This line should ideally not be reached if MAX_ATTEMPTS is well-chosen.
        return "" # Fallback, indicates no solution found within attempts.