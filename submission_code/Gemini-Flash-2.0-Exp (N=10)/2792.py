class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        xor_sum = 0
        for val in derived:
            xor_sum ^= val
        return xor_sum == 0