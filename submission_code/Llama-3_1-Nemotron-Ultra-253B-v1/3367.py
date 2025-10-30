from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        total = 0
        for num in nums:
            s = str(num)
            max_d = max(s)
            encrypted = int(max_d * len(s))
            total += encrypted
        return total