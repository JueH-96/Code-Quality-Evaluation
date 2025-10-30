import collections
from typing import List, Optional, Tuple

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    """
    Given a 0-indexed array of strings words and a character x,
    this class provides a method to find the indices of words 
    that contain the character x.
    """
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        """
        Finds the indices of words in the input list that contain the given character.

        Args:
            words: A list of strings.
            x: The character to search for within the words.

        Returns:
            A list of integers representing the 0-based indices of the words 
            that contain the character x. The order of indices in the returned 
            list does not matter.
        """
        
        # Initialize an empty list to store the indices of matching words.
        result_indices = []
        
        # Iterate through the words list along with their indices.
        # enumerate provides pairs of (index, element).
        for index, word in enumerate(words):
            # Check if the character x is present in the current word.
            # The 'in' operator efficiently checks for substring/character presence.
            if x in word:
                # If the character is found, append the current index to the result list.
                result_indices.append(index)
                
        # Return the list containing the indices of all words that contain x.
        return result_indices

        # --- Alternative concise solution using list comprehension ---
        # This achieves the same result in a single line.
        # return [index for index, word in enumerate(words) if x in word]