class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        len_word1 = len(word1)
        len_word2 = len(word2)
        
        if len_word2 > len_word1:
            return 0
        
        count_word2 = Counter(word2)
        valid_count = 0
        
        for start in range(len_word1):
            count_current = Counter()
            for end in range(start, len_word1):
                count_current[word1[end]] += 1
                
                if end - start + 1 >= len_word2:
                    if all(count_current[char] >= count_word2[char] for char in count_word2):
                        valid_count += 1
        
        return valid_count