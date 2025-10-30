class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        count = Counter(s)
        
        # Initialize the result to the length of the string
        result = len(s)
        
        # Iterate over each character in the string
        for char in count:
            # If the character appears only once, it cannot be removed
            if count[char] == 1:
                result -= 1
        
        # The result cannot be less than 1
        return max(result, 1)