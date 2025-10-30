class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        @lru_cache(None)
        def dp(idx, even_cnt, odd_cnt, rem, tight):
            if idx == len_num:
                return even_cnt == odd_cnt and rem == 0

            ans = 0
            limit = digits[idx] if tight else 9

            for digit in range(limit + 1):
                new_tight = tight and (digit == limit)
                new_rem = (rem * 10 + digit) % k

                if digit % 2 == 0:
                    ans += dp(idx + 1, even_cnt + 1, odd_cnt, new_rem, new_tight)
                else:
                    ans += dp(idx + 1, even_cnt, odd_cnt + 1, new_rem, new_tight)

            return ans

        def count(num):
            nonlocal digits, len_num
            digits = [int(d) for d in str(num)]
            len_num = len(digits)
            return dp(0, 0, 0, 0, True)
        
        
        return count(high) - count(low -1)