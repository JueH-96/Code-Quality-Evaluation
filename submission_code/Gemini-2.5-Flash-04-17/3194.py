from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        """
        Finds the indices of words in a list that contain a given character.

        Args:
            words: A list of strings.
            x: The character to search for.

        Returns:
            A list of indices of words containing the character x.
        """
        result_indices = []
        for i, word in enumerate(words):
            if x in word:
                result_indices.append(i)
        return result_indices