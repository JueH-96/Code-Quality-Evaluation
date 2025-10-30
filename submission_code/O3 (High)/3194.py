from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        """
        Returns a list of indices of the words that contain the character x.

        Args:
        words (List[str]): List of lowercase words.
        x (str): A single lowercase character.

        Returns:
        List[int]: Indices where the character x occurs in the corresponding word.
        """
        # Simple list comprehension to gather indices where x appears in the word
        return [i for i, word in enumerate(words) if x in word]