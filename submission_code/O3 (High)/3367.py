from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for x in nums:
            s = str(x)
            max_digit = max(s)              # largest digit character
            encrypted = int(max_digit * len(s))  # repeat for the length of the number
            total += encrypted
        return total