class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        original = [0] * n
        
        # Compute the original array
        for i in range(n):
            if i == n - 1:
                original[i] = derived[i] ^ original[0]
            else:
                original[i] = derived[i] ^ original[i + 1]
        
        # Check if the original array is valid (contains only 0s and 1s)
        for num in original:
            if num != 0 and num != 1:
                return False
        
        return True