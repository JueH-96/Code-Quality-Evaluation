import functools

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        """
        Calculates the number of beautiful integers in the range [low, high].
        A beautiful integer has an equal number of even and odd digits and is divisible by k.
        """

        # A nested helper function that counts beautiful integers up to a number represented by s_str.
        # This approach encapsulates the DP logic and state for each computation (for high and low-1).
        def solve(s_str: str) -> int:
            n = len(s_str)
            
            # functools.lru_cache is used for memoization to store results of subproblems.
            # The cache is tied to this specific dp_recursive instance, which is created
            # fresh for each call to solve(), so no manual cache clearing is needed.
            @functools.lru_cache(None)
            def dp_recursive(pos: int, rem: int, balance: int, is_tight: bool, is_leading_zero: bool) -> int:
                """
                Calculates the number of ways to complete a beautiful number from the current state.
                
                Args:
                    pos: current digit index (from left, 0-indexed).
                    rem: remainder of the number formed so far modulo k.
                    balance: count of even digits - count of odd digits.
                    is_tight: True if we are restricted by the digits of s_str.
                    is_leading_zero: True if we are currently placing leading zeros.
                
                Returns:
                    The count of valid numbers that can be formed from this state.
                """
                # Base case: We have processed all digits.
                if pos == n:
                    # A valid beautiful number must:
                    # 1. Be non-zero (is_leading_zero is False).
                    # 2. Have equal counts of even and odd digits (balance is 0).
                    # 3. Be divisible by k (rem is 0).
                    return 1 if not is_leading_zero and balance == 0 and rem == 0 else 0

                res = 0
                # Determine the upper bound for the current digit.
                limit = int(s_str[pos]) if is_tight else 9

                for d in range(limit + 1):
                    new_tight = is_tight and (d == limit)
                    
                    if is_leading_zero and d == 0:
                        # Case 1: Still in the leading zero state.
                        # This explores numbers with fewer digits than s_str.
                        # The state for the subproblem (a shorter number) effectively resets.
                        res += dp_recursive(pos + 1, 0, 0, new_tight, True)
                    else:
                        # Case 2: We are placing a regular digit.
                        new_rem = (rem * 10 + d) % k
                        # Update balance: +1 for an even digit, -1 for an odd digit.
                        new_balance = balance + (1 if d % 2 == 0 else -1)
                        
                        # Recurse for the next position.
                        res += dp_recursive(pos + 1, new_rem, new_balance, new_tight, False)
                
                return res

            # Initial call to start the DP process.
            # Start at pos 0, rem 0, balance 0, with tight constraint, in leading zero state.
            return dp_recursive(0, 0, 0, True, True)

        # The number of beautiful integers in [low, high] is count(high) - count(low - 1).
        count_high = solve(str(high))
        count_low = solve(str(low - 1))
        
        return count_high - count_low