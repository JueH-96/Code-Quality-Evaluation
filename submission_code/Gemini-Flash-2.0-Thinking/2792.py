class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        xor_sum_derived = 0
        for val in derived:
            xor_sum_derived ^= val
        return xor_sum_derived == 0