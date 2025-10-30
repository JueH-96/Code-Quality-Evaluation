class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # We want to check if the XOR of all derived values equals 0.
        # If it does, a valid original array can exist; otherwise, it cannot.
        
        total_xor = 0
        for val in derived:
            total_xor ^= val
        
        return (total_xor == 0)