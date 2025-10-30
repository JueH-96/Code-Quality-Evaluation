class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        nums = [num1, num2, num3]
        padded_nums = [str(num).zfill(4) for num in nums]
        key_str = ""
        for i in range(4):
            min_digit = min(int(num[i]) for num in padded_nums)
            key_str += str(min_digit)
        key = int(key_str)
        return key