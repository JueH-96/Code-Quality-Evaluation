class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        # Try starting with original[0] = 0
        original = [0] * n
        
        # Fill in the rest of original array based on derived values
        for i in range(n-1):
            original[i+1] = original[i] ^ derived[i]
            
        # Check if last XOR matches with first element
        if (original[n-1] ^ original[0]) == derived[n-1]:
            return True
            
        # Try starting with original[0] = 1
        original = [1] * 1 + [0] * (n-1)
        
        # Fill in the rest of original array based on derived values
        for i in range(n-1):
            original[i+1] = original[i] ^ derived[i]
            
        # Check if last XOR matches with first element
        if (original[n-1] ^ original[0]) == derived[n-1]:
            return True
            
        return False