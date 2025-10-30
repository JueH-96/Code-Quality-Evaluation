from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        """
        Finds the indices of words in a list that contain a specific character.

        Args:
            words: A 0-indexed array of strings.
            x: A character to search for within the words.

        Returns:
            A list of integers representing the 0-indexed positions of words
            that contain the character x. The order of indices in the returned
            array does not matter.
        """
        result_indices = []
        for i, word in enumerate(words):
            # Check if the character x is present in the current word
            if x in word:
                result_indices.append(i)
        return result_indices