class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # Calculate the XOR of all elements in the derived array
        xor_sum = 0
        for num in derived:
            xor_sum ^= num
        
        # If the XOR of all elements is 0, then a valid original array exists
        return xor_sum == 0