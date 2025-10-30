class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        memo = {}
        
        def dp(rem_aa, rem_bb, rem_ab, last):
            if rem_aa == 0 and rem_bb == 0 and rem_ab == 0:
                return 0
            state = (rem_aa, rem_bb, rem_ab, last)
            if state in memo:
                return memo[state]
            ans = 0
            if last == 0:  # no last block
                if rem_aa > 0:
                    ans = max(ans, 1 + dp(rem_aa - 1, rem_bb, rem_ab, 1))
                if rem_bb > 0:
                    ans = max(ans, 1 + dp(rem_aa, rem_bb - 1, rem_ab, 2))
                if rem_ab > 0:
                    ans = max(ans, 1 + dp(rem_aa, rem_bb, rem_ab - 1, 3))
            elif last == 1:  # last was "AA"
                if rem_bb > 0:  # can place "BB"
                    ans = max(ans, 1 + dp(rem_aa, rem_bb - 1, rem_ab, 2))
            elif last == 2:  # last was "BB"
                if rem_aa > 0:
                    ans = max(ans, 1 + dp(rem_aa - 1, rem_bb, rem_ab, 1))
                if rem_ab > 0:
                    ans = max(ans, 1 + dp(rem_aa, rem_bb, rem_ab - 1, 3))
            elif last == 3:  # last was "AB"
                if rem_aa > 0:
                    ans = max(ans, 1 + dp(rem_aa - 1, rem_bb, rem_ab, 1))
                if rem_ab > 0:
                    ans = max(ans, 1 + dp(rem_aa, rem_bb, rem_ab - 1, 3))
            memo[state] = ans
            return ans
        
        max_blocks = dp(x, y, z, 0)
        return 2 * max_blocks