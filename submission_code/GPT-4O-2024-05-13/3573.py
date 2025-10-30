from collections import Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        def is_valid(sub_counter, word2_counter):
            for char in word2_counter:
                if sub_counter[char] < word2_counter[char]:
                    return False
            return True
        
        word2_counter = Counter(word2)
        word2_length = len(word2)
        word1_length = len(word1)
        count = 0
        
        for i in range(word1_length):
            sub_counter = Counter()
            for j in range(i, word1_length):
                sub_counter[word1[j]] += 1
                if j - i + 1 >= word2_length and is_valid(sub_counter, word2_counter):
                    count += 1
        
        return count