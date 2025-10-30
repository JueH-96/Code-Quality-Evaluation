class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        time = 0
        
        while True:
            # Remove the first k characters
            word = word[k:]
            
            # Add k characters to the end
            word += word[:k]
            
            time += 1
            
            # Check if the word has reverted to its initial state
            if word == word[:n]:
                return time