class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def is_symmetric(n: int) -> bool:
            s = str(n)
            length = len(s)
            if length % 2 != 0:
                return False
            mid = length // 2
            sum1 = sum(int(digit) for digit in s[:mid])
            sum2 = sum(int(digit) for digit in s[mid:])
            return sum1 == sum2

        count = 0
        for i in range(low, high + 1):
            if is_symmetric(i):
                count += 1
        return count