class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_up_to(n, k):
            if n < 10:
                return 0
            s = str(n)
            max_length = len(s)
            memo = {}
            
            def dp(pos, diff, remainder, is_tight, is_leading_zero):
                if pos == max_length:
                    if not is_leading_zero and diff == 0 and remainder == 0:
                        return 1
                    return 0
                key = (pos, diff + 5, remainder, is_tight, is_leading_zero)
                if key in memo:
                    return memo[key]
                digit_limit = int(s[pos]) if is_tight else 9
                total = 0
                for d in range(0, digit_limit + 1):
                    new_diff = diff + 1 if d % 2 == 0 else diff - 1
                    if is_leading_zero:
                        if d == 0:
                            total += dp(pos + 1, new_diff, (remainder * 10 + d) % k, is_tight and d == digit_limit, True)
                        else:
                            new_is_tight = is_tight and d == digit_limit
                            new_is_leading_zero = False
                            total += dp(pos + 1, new_diff, (remainder * 10 + d) % k, new_is_tight, new_is_leading_zero)
                    else:
                        new_is_tight = is_tight and d == digit_limit
                        new_is_leading_zero = False
                        total += dp(pos + 1, new_diff, (remainder * 10 + d) % k, new_is_tight, new_is_leading_zero)
                memo[key] = total
                return total
            return dp(0, 0, 0, True, True)
        
        return count_up_to(high, k) - count_up_to(max(0, low - 1), k)