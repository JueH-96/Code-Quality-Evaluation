class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for i in range(low, high + 1):
            str_i = str(i)
            if len(str_i) % 2 == 0:
                half = len(str_i) // 2
                if sum(int(digit) for digit in str_i[:half]) == sum(int(digit) for digit in str_i[half:]):
                    count += 1
        return count