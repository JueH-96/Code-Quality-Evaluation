class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        if n == 1:
            return derived[0] == 0
        
        # Start with two possible values for original[0]
        possible_starts = [0, 1]
        
        for start in possible_starts:
            original = [0] * n
            original[0] = start
            valid = True
            
            # Build the original array based on derived
            for i in range(n - 1):
                original[i + 1] = derived[i] ^ original[i]
            
            # Check the last condition
            if derived[n - 1] != (original[n - 1] ^ original[0]):
                valid = False
            
            if valid:
                return True
        
        return False