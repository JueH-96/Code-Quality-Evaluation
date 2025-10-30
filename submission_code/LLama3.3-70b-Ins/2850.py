class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # Initialize the maximum length of the new string
        max_length = 0
        
        # Iterate over all possible combinations of x, y, and z
        for i in range(x + 1):
            for j in range(y + 1):
                for k in range(z + 1):
                    # Construct the new string
                    new_string = 'A' * (2 * i) + 'B' * (2 * j) + 'AB' * k
                    
                    # Check if the new string contains "AAA" or "BBB" as a substring
                    if 'AAA' not in new_string and 'BBB' not in new_string:
                        # Update the maximum length
                        max_length = max(max_length, len(new_string))
        
        # Return the maximum possible length of the new string
        return max_length