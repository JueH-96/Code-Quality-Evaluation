from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        def encrypt(x: int) -> int:
            return int(str(max(str(x))) * len(str(x)))
        
        return sum(encrypt(num) for num in nums)