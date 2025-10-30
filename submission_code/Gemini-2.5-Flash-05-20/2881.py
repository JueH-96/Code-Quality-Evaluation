from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        # Initialize an empty list to store all the valid split parts.
        result = []
        
        # Iterate through each word in the input list 'words'.
        for word in words:
            # Split the current word by the given separator.
            # The .split() method returns a list of substrings.
            # It handles cases like leading/trailing separators or consecutive separators
            # by producing empty strings, which we will filter out.
            split_parts = word.split(separator)
            
            # Iterate through the parts obtained from the split.
            for part in split_parts:
                # Check if the part is not an empty string.
                # An empty string evaluates to False in a boolean context,
                # so 'if part:' is a concise way to check if 'part' is non-empty.
                if part:
                    # If the part is not empty, add it to our result list.
                    result.append(part)
                    
        # After processing all words and their split parts, return the accumulated result.
        return result