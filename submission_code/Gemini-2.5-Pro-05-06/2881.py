from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        """
        Splits each string in words by separator and returns an array of strings 
        containing the new strings formed after the splits, excluding empty strings.
        """
        
        result_strings = []  # This list will store all non-empty parts after splitting.
        
        # Iterate through each word in the input list 'words'.
        for word in words:
            # Split the current 'word' using the given 'separator'.
            # The str.split() method returns a list of substrings.
            # For example:
            # "one.two.three".split('.') results in ['one', 'two', 'three'].
            # "$easy$".split('$') results in ['', 'easy', '']. (Empty strings are produced for splits at ends)
            # "|||".split('|') results in ['', '', '', '']. (Empty strings for consecutive separators)
            split_parts = word.split(separator)
            
            # Filter out any empty strings from 'split_parts' and add the
            # non-empty ones to 'result_strings'.
            # A list comprehension '[p for p in split_parts if p]' achieves this:
            # 'p for p in split_parts' iterates through each part.
            # 'if p' checks if the part 'p' is non-empty (an empty string evaluates to False).
            # The result is a new list containing only non-empty parts.
            #
            # 'result_strings.extend(...)' adds all elements from this new list
            # to 'result_strings'. If the new list is empty (e.g., all parts were empty),
            # extend([]) does nothing, which is the correct behavior.
            result_strings.extend([p for p in split_parts if p])
            
        # Return the final list of non-empty strings, in the order they were formed.
        return result_strings