class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        # Convert the string to lowercase and take unique characters
        unique_lower_chars = set(word.lower())
        count = 0
        
        # For each unique character (in lowercase),
        # check if it appears in both lowercase and uppercase in the original word
        for ch in unique_lower_chars:
            if ch in word and ch.upper() in word:
                count += 1
        
        return count