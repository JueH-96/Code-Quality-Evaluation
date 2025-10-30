from collections import Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # Count the frequency of each character in word2
        word2_counter = Counter(word2)
        word2_len = len(word2)
        valid_count = 0
        
        # Iterate over all possible substrings of word1
        for i in range(len(word1)):
            for j in range(i + word2_len, len(word1) + 1):
                # Check if the substring can be rearranged to have word2 as a prefix
                if not (Counter(word1[i:j]) - word2_counter):
                    valid_count += 1
        
        return valid_count