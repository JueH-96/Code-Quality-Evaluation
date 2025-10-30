class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        # Collect all lowercase letters and uppercase letters (mapped to lowercase)
        lowers = {ch for ch in word if ch.islower()}
        uppers_mapped = {ch.lower() for ch in word if ch.isupper()}
        
        # The special letters are those present in both sets
        return len(lowers & uppers_mapped)