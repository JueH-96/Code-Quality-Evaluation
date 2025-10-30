from typing import List

class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        return sum(int(max(s) * len(s)) for s in map(str, nums))