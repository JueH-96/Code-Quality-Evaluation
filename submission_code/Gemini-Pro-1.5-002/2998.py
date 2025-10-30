class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s_num = str(num)
            n = len(s_num)
            if n % 2 == 0:
                n //= 2
                sum1 = sum(int(digit) for digit in s_num[:n])
                sum2 = sum(int(digit) for digit in s_num[n:])
                if sum1 == sum2:
                    count += 1
        return count