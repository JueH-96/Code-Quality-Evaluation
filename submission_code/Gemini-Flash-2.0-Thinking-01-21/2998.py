class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s_num = str(num)
            n = len(s_num)
            if n % 2 == 0:
                mid = n // 2
                sum_first_half = 0
                sum_second_half = 0
                for i in range(mid):
                    sum_first_half += int(s_num[i])
                for i in range(mid, n):
                    sum_second_half += int(s_num[i])
                if sum_first_half == sum_second_half:
                    count += 1
        return count