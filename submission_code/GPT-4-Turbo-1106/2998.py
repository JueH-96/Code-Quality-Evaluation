class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            str_x = str(x)
            length = len(str_x)
            if length % 2 == 0:
                mid = length // 2
                if sum(int(digit) for digit in str_x[:mid]) == sum(int(digit) for digit in str_x[mid:]):
                    count += 1
        return count