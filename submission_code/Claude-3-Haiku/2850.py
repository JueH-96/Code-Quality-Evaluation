class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # Initialize the maximum length to 0
        max_length = 0
        
        # Iterate through all possible combinations of strings
        for i in range(x+1):
            for j in range(y+1):
                for k in range(z+1):
                    # Construct the current string
                    current_string = "AA" * i + "BB" * j + "AB" * k
                    
                    # Check if the current string does not contain "AAA" or "BBB" as a substring
                    if "AAA" not in current_string and "BBB" not in current_string:
                        # Update the maximum length if the current string is longer
                        max_length = max(max_length, len(current_string))
        
        return max_length