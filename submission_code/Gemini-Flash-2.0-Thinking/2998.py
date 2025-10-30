class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s_num = str(num)
            n = len(s_num)
            if n % 2 == 0:
                half = n // 2
                first_half_sum = sum(int(digit) for digit in s_num[:half])
                second_half_sum = sum(int(digit) for digit in s_num[half:])
                if first_half_sum == second_half_sum:
                    count += 1
        return count