class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        # Initialize an empty list to store the indices of words containing x
        result = []
        
        # Iterate over each word with its index
        for idx, word in enumerate(words):
            # Check if the character x is in the current word
            if x in word:
                # If yes, append the index to our result list
                result.append(idx)
        
        # Return the list of indices
        return result