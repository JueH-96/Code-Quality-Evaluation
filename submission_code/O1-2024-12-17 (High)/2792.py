class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        
        # We'll try to fix original[0] = 0 and original[0] = 1.
        # If either assumption leads to a consistent solution, return True.
        # Otherwise, return False.
        
        def can_form_original(first_val: int) -> bool:
            # "x" will track the current value of original[i]
            x = first_val
            # Derive all values up to original[n-1] using derived
            for i in range(n - 1):
                x = x ^ derived[i]  # x_(i+1) = x_i ^ derived[i]
            
            # Now, x holds original[n-1]
            # Check consistency of the last relation: original[n-1] ^ original[0] == derived[n-1]
            return (x ^ first_val) == derived[-1]
        
        # Try original[0] = 0
        if can_form_original(0):
            return True
        # Try original[0] = 1
        if can_form_original(1):
            return True
        
        return False