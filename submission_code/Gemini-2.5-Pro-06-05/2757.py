class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def get_count_le(s: str, limit: int) -> int:
            """
            Counts the number of integers x in [1, s] such that the sum of their digits is at most `limit`.
            This is achieved by counting numbers in [0, s] and subtracting 1 (for the number 0).
            """
            if limit < 0:
                return 0
            
            n = len(s)
            memo = {}

            def dp(index: int, current_sum: int, is_tight: bool) -> int:
                """
                A recursive function with memoization (Digit DP) to count numbers.
                It returns the count of ways to form a number from the current state
                such that the total digit sum does not exceed `limit`.
                
                - index: The current digit position to fill.
                - current_sum: The sum of digits placed so far.
                - is_tight: A boolean indicating if we are restricted by the digits of `s`.
                """
                if current_sum > limit:
                    return 0
                if index == n:
                    return 1

                state = (index, current_sum, is_tight)
                if state in memo:
                    return memo[state]

                res = 0
                upper_bound = int(s[index]) if is_tight else 9

                for digit in range(upper_bound + 1):
                    new_tight = is_tight and (digit == upper_bound)
                    res = (res + dp(index + 1, current_sum + digit, new_tight)) % MOD
                
                memo[state] = res
                return res
            
            # The initial call to dp counts numbers in [0, s] with the given property.
            # We subtract 1 to exclude the number 0 from our count.
            count_0_to_s = dp(0, 0, True)
            return (count_0_to_s - 1 + MOD) % MOD

        # Calculate the count of good integers up to num2.
        # This is count(sum <= max_sum) - count(sum <= min_sum - 1).
        ans_up_to_num2 = (get_count_le(num2, max_sum) - get_count_le(num2, min_sum - 1) + MOD) % MOD
        
        # Calculate the count of good integers up to num1 - 1.
        num1_minus_1 = str(int(num1) - 1)
        ans_up_to_num1_minus_1 = (get_count_le(num1_minus_1, max_sum) - get_count_le(num1_minus_1, min_sum - 1) + MOD) % MOD
        
        # The final answer is the difference between the two counts.
        return (ans_up_to_num2 - ans_up_to_num1_minus_1 + MOD) % MOD