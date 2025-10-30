class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        xor_sum = 0
        
        for i in range(n):
            xor_sum ^= derived[i]
        
        return xor_sum == 0