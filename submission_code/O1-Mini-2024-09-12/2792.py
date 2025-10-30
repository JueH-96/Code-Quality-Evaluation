class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        total_xor = 0
        for num in derived:
            total_xor ^= num
        return total_xor == 0