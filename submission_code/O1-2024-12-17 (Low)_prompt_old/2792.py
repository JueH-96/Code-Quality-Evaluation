class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # The key insight is that for the circular XOR pattern to hold,
        # the XOR of all elements in 'derived' must be 0.
        # Reasoning:
        # If we let original[0] = x (either 0 or 1), then each subsequent original[i]
        # is uniquely determined by derived[i-1] and original[i-1]. The last condition
        # derived[n-1] = original[n-1] ^ original[0] forces the XOR of all derived
        # elements to be 0 in order for a consistent assignment to exist.
        
        xor_sum = 0
        for val in derived:
            xor_sum ^= val
        return (xor_sum == 0)