class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        
        # We'll try both possible values for original[0]: 0 or 1
        for first_val in (0, 1):
            original = [first_val]
            
            # Build the original array based on the derived array
            for i in range(n - 1):
                original.append(original[-1] ^ derived[i])
            
            # Check if the last XOR condition holds
            if (original[-1] ^ original[0]) == derived[-1]:
                return True
        
        return False