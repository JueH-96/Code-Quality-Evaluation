from collections import Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        word2_len = len(word2)
        word1_len = len(word1)
        word2_counter = Counter(word2)
        total_count = 0
        
        for i in range(word1_len):
            current_counter = Counter()
            for j in range(i, word1_len):
                current_counter[word1[j]] += 1
                if self.is_valid(current_counter, word2_counter):
                    total_count += 1
                else:
                    break
        return total_count
    
    def is_valid(self, current_counter, word2_counter):
        for char, count in word2_counter.items():
            if current_counter[char] < count:
                return False
        return True