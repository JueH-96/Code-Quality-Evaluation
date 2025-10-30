class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for i in range(low, high + 1):
            s = str(i)
            if len(s) % 2 == 0:
                n = len(s) // 2
                sum_first = sum(int(digit) for digit in s[:n])
                sum_last = sum(int(digit) for digit in s[n:])
                if sum_first == sum_last:
                    count += 1
        return count