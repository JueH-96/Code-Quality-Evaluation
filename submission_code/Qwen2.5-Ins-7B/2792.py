class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        original = [0] * n
        original[0] = 0
        
        for i in range(n - 1):
            if original[i] == derived[i]:
                original[i + 1] = original[i]
            else:
                original[i + 1] = original[i] ^ 1
        
        if original[n - 1] ^ original[0] == derived[n - 1]:
            return True
        return False