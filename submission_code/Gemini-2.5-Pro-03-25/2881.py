import collections
from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        """
        Splits each string in the words list by the given separator and returns
        a flattened list of non-empty resulting strings, maintaining order.

        Args:
            words: A list of strings.
            separator: The character to split by.

        Returns:
            A list of strings containing the non-empty parts after splitting,
            in the order they appeared.
        """
        
        result = [] # Initialize an empty list to store the final split strings
        
        # Iterate through each word in the input list 'words'
        for word in words:
            # Split the current word using the specified separator.
            # The split() method returns a list of substrings.
            parts = word.split(separator)
            
            # Iterate through the parts obtained from the split
            for part in parts:
                # Check if the part is not an empty string.
                # An empty string evaluates to False in a boolean context.
                if part: 
                    # If the part is not empty, add it to the result list.
                    result.append(part)
                    
        # Return the final list containing all non-empty split parts.
        return result