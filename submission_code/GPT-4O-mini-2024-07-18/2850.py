class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # Calculate the maximum length of the string we can form
        # We can use a greedy approach to maximize the length while avoiding "AAA" and "BBB"
        
        # Start with the total length of the strings we have
        total_length = 0
        
        # While we have strings to use
        while x > 0 or y > 0 or z > 0:
            if y > x:  # Prefer to use "BB" if there are more "BB" than "AA"
                if y > 0:
                    total_length += 2  # Add "BB"
                    y -= 1
                if z > 0:  # Then add "AB" if available
                    total_length += 2  # Add "AB"
                    z -= 1
            else:  # Prefer to use "AA" if there are more "AA" or equal
                if x > 0:
                    total_length += 2  # Add "AA"
                    x -= 1
                if z > 0:  # Then add "AB" if available
                    total_length += 2  # Add "AB"
                    z -= 1
            
            # If we can't add "AB" anymore, we should add the remaining "AA" or "BB"
            if z == 0:
                if x > 0:
                    total_length += 2  # Add remaining "AA"
                    x -= 1
                elif y > 0:
                    total_length += 2  # Add remaining "BB"
                    y -= 1
        
        return total_length