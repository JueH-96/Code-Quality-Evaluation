from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        """
        This function finds the indices of words in a given list that contain a specific character.

        Args:
        words (List[str]): A list of strings.
        x (str): A character to search for in the words.

        Returns:
        List[int]: A list of indices of words that contain the character x.
        """
        
        # Initialize an empty list to store the indices of words containing the character x
        indices = []
        
        # Iterate over the list of words with their indices
        for i, word in enumerate(words):
            # Check if the character x is in the current word
            if x in word:
                # If the character is found, append the index of the word to the list of indices
                indices.append(i)
        
        # Return the list of indices
        return indices