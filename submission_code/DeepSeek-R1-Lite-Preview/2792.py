class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        xor_sum = 0
        for value in derived:
            xor_sum ^= value
        return xor_sum == 0