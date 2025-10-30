class Solution:
    def smallestNumber(self, n: int) -> int:
        # Convert the input number to binary string
        binary_str = bin(n)[2:]
        
        # Find the index of the rightmost 0 in the binary string
        right_zero_idx = binary_str.rfind('0')
        
        # If there are no 0s, return the input number
        if right_zero_idx == -1:
            return n
        
        # Replace the rightmost 0 with 1
        new_binary_str = binary_str[:right_zero_idx] + '1' + binary_str[right_zero_idx+1:]
        
        # Convert the new binary string back to an integer and return it
        return int(new_binary_str, 2)