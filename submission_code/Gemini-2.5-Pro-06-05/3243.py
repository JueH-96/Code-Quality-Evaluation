class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        
        s_val = int(s)
        p_len_base = 10**len(s)

        def count_le(N_str: str) -> int:
            """
            Counts non-negative integers <= int(N_str) with all digits <= limit
            using digit DP.
            """
            memo = {}  # Memoization table for this specific call
            n = len(N_str)

            def dp(idx: int, is_tight: bool) -> int:
                """
                dp(idx, is_tight) returns the number of ways to complete a valid
                number from the current state.
                - idx: current digit position (from left)
                - is_tight: True if we are restricted by the digits of N_str
                """
                if idx == n:
                    return 1  # Successfully formed one valid number
                
                state = (idx, is_tight)
                if state in memo:
                    return memo[state]
                
                res = 0
                # The upper bound for the current digit
                ub = int(N_str[idx]) if is_tight else 9
                
                # Iterate through possible digits d for the current position
                for d in range(ub + 1):
                    if d > limit:
                        break  # Digits cannot exceed the limit
                    
                    # If we use a digit smaller than the tight bound, the next
                    # states are no longer tight.
                    new_tight = is_tight and (d == ub)
                    res += dp(idx + 1, new_tight)
                
                memo[state] = res
                return res

            return dp(0, True)

        def get_count(N: int) -> int:
            """
            Calculates the number of powerful integers <= N.
            """
            if N < s_val:
                return 0
            
            p_upper_bound = (N - s_val) // p_len_base
            
            # count_le will count the number of valid prefixes `p`
            # from 0 to p_upper_bound.
            return count_le(str(p_upper_bound))

        # The result is the number of powerful integers <= finish minus those <= start - 1.
        return get_count(finish) - get_count(start - 1)