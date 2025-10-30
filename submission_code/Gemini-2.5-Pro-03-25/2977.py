import sys
# Assuming List is imported from typing if type hints are strictly needed, 
# but it's often implicit in LeetCode-like environments.
# from typing import List 

class Solution:
    """
    Determines if a string 's' is an acronym of a list of strings 'words'.
    
    An acronym is formed by concatenating the first character of each string 
    in 'words' in order.
    """
    def isAcronym(self, words: list[str], s: str) -> bool:
        """
        Checks if s is the acronym of words.

        Args:
            words: A list of strings.
            s: The potential acronym string.

        Returns:
            True if s is the acronym of words, False otherwise.
        """
        
        # Optimization: If the number of words does not equal the length of s,
        # it's impossible for s to be the acronym.
        if len(words) != len(s):
            return False
            
        # Construct the potential acronym by iterating through the words list
        # and taking the first character of each word.
        # The constraints guarantee words[i].length >= 1, so words[i][0] is safe.
        built_acronym = ""
        for word in words:
            # Since constraints guarantee word length >= 1, we don't need to check
            # for empty strings.
            built_acronym += word[0]
            
        # Compare the constructed acronym with the given string s.
        return built_acronym == s

        # Alternative Pythonic way using list comprehension and join:
        # if len(words) != len(s):
        #     return False
        # return "".join(word[0] for word in words) == s