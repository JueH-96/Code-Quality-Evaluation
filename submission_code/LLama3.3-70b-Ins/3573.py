from collections import Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        count = 0
        word2_count = Counter(word2)
        
        for i in range(len(word1)):
            for j in range(i + 1, len(word1) + 1):
                substring = word1[i:j]
                substring_count = Counter(substring)
                
                # Check if the substring can be rearranged to have word2 as a prefix
                if all(substring_count[char] >= word2_count[char] for char in word2_count):
                    count += 1
                    
        return count