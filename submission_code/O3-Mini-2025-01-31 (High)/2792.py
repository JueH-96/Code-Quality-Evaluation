from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        xor_sum = 0
        for d in derived:
            xor_sum ^= d
        return xor_sum == 0