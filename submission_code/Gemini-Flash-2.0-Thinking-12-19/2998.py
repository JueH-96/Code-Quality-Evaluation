class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s_num = str(num)
            n = len(s_num)
            if n % 2 == 0:
                half_n = n // 2
                first_half = s_num[:half_n]
                second_half = s_num[half_n:]
                sum_first_half = 0
                sum_second_half = 0
                for digit in first_half:
                    sum_first_half += int(digit)
                for digit in second_half:
                    sum_second_half += int(digit)
                if sum_first_half == sum_second_half:
                    count += 1
        return count