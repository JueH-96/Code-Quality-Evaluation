from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        """
        Finds the indices of words in a list that contain a specific character.

        Args:
            words: A list of strings.
            x: The character to search for.

        Returns:
            A list of indices of the words containing the character x.
        """
        
        # Using a list comprehension for a concise and Pythonic solution.
        # We iterate through the 'words' list along with their indices using enumerate().
        # For each index 'i' and its corresponding 'word', we check if the character 'x'
        # is present in the 'word'. If it is, the index 'i' is included in the
        # resulting list.
        return [i for i, word in enumerate(words) if x in word]