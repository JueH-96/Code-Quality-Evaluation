class Solution:
    def countKeyChanges(self, s: str) -> int:
        # Convert the string to lowercase to ignore case differences
        s_lower = s.lower()
        # Initialize the counter for key changes
        counter = 0
        # Iterate through the string starting from the second character
        for i in range(1, len(s_lower)):
            if s_lower[i] != s_lower[i - 1]:
                counter += 1
        # Return the total count of key changes
        return counter