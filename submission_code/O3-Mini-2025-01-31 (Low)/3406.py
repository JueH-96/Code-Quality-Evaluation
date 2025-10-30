class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7

        # Using dp recursion with memoization.
        # dp(z, o, last, run) returns the number of valid arrays that can be formed
        # with z zeros and o ones remaining, where the last placed element is 'last'
        # (0 or 1) with current consecutive run length 'run'. If last is None, we haven't placed any element.
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(z, o, last, run):
            # Base case: if no remaining symbols, one valid array.
            if z == 0 and o == 0:
                return 1

            total = 0
            # Option to add a 0:
            if z > 0:
                if last == 0:  # continuing run of 0's
                    if run < limit:
                        total += dp(z - 1, o, 0, run + 1)
                        total %= MOD
                else:
                    # either last is 1 or None, so new run starting with 0 is safe.
                    total += dp(z - 1, o, 0, 1)
                    total %= MOD
            # Option to add a 1:
            if o > 0:
                if last == 1:  # continuing run of 1's
                    if run < limit:
                        total += dp(z, o - 1, 1, run + 1)
                        total %= MOD
                else:
                    total += dp(z, o - 1, 1, 1)
                    total %= MOD
            return total % MOD

        return dp(zero, one, None, 0)
  
# Example usage:
# sol = Solution()
# print(sol.numberOfStableArrays(1, 1, 2))  # Expected output: 2
# print(sol.numberOfStableArrays(1, 2, 1))  # Expected output: 1
# print(sol.numberOfStableArrays(3, 3, 2))  # Expected output: 14