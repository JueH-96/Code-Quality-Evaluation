from collections import Counter

class Solution:
    def makeStringGood(self, s: str) -> int:
        # Count the frequency of each character in the string
        char_counts = Counter(s)
        
        # Initialize the number of operations required to 0
        operations = 0
        
        # Iterate through the characters in the string
        for char, count in char_counts.items():
            # If the character does not occur the same number of times as other characters
            if count != min(char_counts.values()):
                # Calculate the number of operations required to make the character occur the same number of times
                operations += count - min(char_counts.values())
        
        return operations