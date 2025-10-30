class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        
        @cache
        def check(i, sum_, odd_digit_num, even_digit_num, is_limit, has_nonzero, sign):
            if not has_nonzero and sign == 0:
                return 0
            if i == len():
                return int(odd_digit_num == even_digit_num and sum_ % k == 0)

            res = 0
            if not is_limit:
                res = check(i + 1, sum_, 0, 0, False, False, -1)
            
            up = nums[i] if is_limit else 9
            for d in range(up + 1):
                res += check(i + 1, (sum_ * 10 + d) % k, odd_digit_num + int(d % 2 == 1), even_digit_num + int(d % 2 == 0), is_limit and up == d, has_nonzero or d > 0, sign)
            return res
        
        len_1, len_2 = len(str(low)), len(str(high))
        nums_1, nums_2 = [int(x) for x in str(low)], [int(x) for x in str(high)]

        res = check(0, 0, 0, 0, True, False, 1) # upper bound
        res -= check(0, 0, 0, 0, True, False, -1) # lower bound - 1
        return res