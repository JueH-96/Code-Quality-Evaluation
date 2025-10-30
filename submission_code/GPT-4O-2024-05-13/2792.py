class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        # XOR of all elements in derived
        xor_sum = 0
        for num in derived:
            xor_sum ^= num
        
        # If the XOR sum of all elements in derived is 0, then a valid original array exists
        return xor_sum == 0