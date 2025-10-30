from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        """
        Finds all indices of words in a list that contain a specific character.

        Args:
            words: A 0-indexed list of strings.
            x: A character to search for in the words.

        Returns:
            A list of indices representing the words that contain the character x.
            The order of indices in the returned list does not matter.
        """
        
        # Initialize an empty list to store the indices of words containing x.
        result_indices = []
        
        # Iterate through the 'words' list using enumerate to get both the index (i)
        # and the word itself (word) at each step.
        for i, word in enumerate(words):
            # Check if the character 'x' is present in the current 'word'.
            # The 'in' operator in Python efficiently checks for substring presence.
            if x in word:
                # If 'x' is found in 'word', add its index 'i' to the result list.
                result_indices.append(i)
                
        # Return the list of collected indices.
        return result_indices

# Alternatively, a more concise list comprehension can be used:
# class Solution:
#     def findWordsContaining(self, words: List[str], x: str) -> List[int]:
#         return [i for i, word in enumerate(words) if x in word]