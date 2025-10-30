class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            num_str = str(num)
            if len(num_str) % 2 == 0:
                mid = len(num_str) // 2
                first_half = sum(int(digit) for digit in num_str[:mid])
                second_half = sum(int(digit) for digit in num_str[mid:])
                if first_half == second_half:
                    count += 1
        return count