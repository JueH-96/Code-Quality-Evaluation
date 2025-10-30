from collections import Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        def is_valid(sub: str, word2_counter: Counter) -> bool:
            sub_counter = Counter(sub)
            for char, count in word2_counter.items():
                if sub_counter[char] < count:
                    return False
            return True
        
        word2_counter = Counter(word2)
        n, m = len(word1), len(word2)
        valid_count = 0
        
        for i in range(n):
            for j in range(i + m, n + 1):
                if is_valid(word1[i:j], word2_counter):
                    valid_count += 1
        
        return valid_count