class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        count = 0
        w2_count = Counter(word2)
        w1_count = Counter()
        
        for i in range(len(word1)):
            w1_count[word1[i]] += 1
            if i >= len(word2):
                w1_count[word1[i - len(word2)]] -= 1
                if w1_count[word1[i - len(word2)]] == 0:
                    del w1_count[word1[i - len(word2)]]
            
            if w1_count == w2_count:
                count += 1
        
        return count