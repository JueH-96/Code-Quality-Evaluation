class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s = str(num)
            if len(s) % 2 == 0:  # must have an even number of digits
                mid = len(s) // 2
                left_sum = sum(int(digit) for digit in s[:mid])
                right_sum = sum(int(digit) for digit in s[mid:])
                if left_sum == right_sum:
                    count += 1
        return count