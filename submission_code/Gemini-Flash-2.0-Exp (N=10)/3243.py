class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count_powerful(num_str, limit, s):
            n = len(num_str)
            m = len(s)
            
            if n < m:
                return 0

            dp = {}
            
            def solve(idx, is_prefix_match, is_limit_reached):
                if idx == n:
                    return is_prefix_match
                
                if (idx, is_prefix_match, is_limit_reached) in dp:
                    return dp[(idx, is_prefix_match, is_limit_reached)]
                
                ans = 0
                upper_bound = int(num_str[idx]) if is_limit_reached else limit
                
                for digit in range(upper_bound + 1):
                    new_prefix_match = is_prefix_match
                    if is_prefix_match:
                        if idx >= n - m:
                            if str(digit) != s[idx - (n - m)]:
                                new_prefix_match = False
                    
                    ans += solve(idx + 1, new_prefix_match, is_limit_reached and (digit == upper_bound))
                
                dp[(idx, is_prefix_match, is_limit_reached)] = ans
                return ans

            return solve(0, True, True)

        return count_powerful(str(finish), limit, s) - count_powerful(str(start - 1), limit, s)