class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        if n == 1:
            return derived[0] == 0
        xor_total = 0
        for num in derived:
            xor_total ^= num
        return xor_total == 0