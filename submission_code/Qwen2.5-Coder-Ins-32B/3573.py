class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        target_count = Counter(word2)
        k = len(word2)
        count = 0
        current_count = Counter()
        
        for i in range(len(word1)):
            current_count[word1[i]] += 1
            
            if i >= k:
                if current_count[word1[i - k]] == 1:
                    del current_count[word1[i - k]]
                else:
                    current_count[word1[i - k]] -= 1
            
            if current_count & target_count == target_count:
                count += len(word1) - i
        
        return count