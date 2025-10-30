class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for x in nums:
            digits = [int(d) for d in str(x)]
            max_digit = max(digits)
            new_num = int(str(max_digit) * len(digits))
            total += new_num
        return total