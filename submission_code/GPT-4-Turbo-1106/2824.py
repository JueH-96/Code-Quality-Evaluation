class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated_str = str(n) + str(n * 2) + str(n * 3)
        return len(concatenated_str) == 9 and all(str(digit) in concatenated_str for digit in range(1, 10))