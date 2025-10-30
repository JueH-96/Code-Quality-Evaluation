class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        len1, len2 = len(word1), len(word2)
        count1, count2 = Counter(word1), Counter(word2)
        
        valid_count = 0
        for i in range(len1 - len2 + 1):
            if word1[i:i+len2] == word2:
                valid_count += 1
            else:
                for j in range(i, i+len2):
                    count1[word1[j]] -= 1
                    if count1[word1[j]] < count2[word1[j]]:
                        break
                else:
                    valid_count += 1
                for j in range(i, i+len2):
                    count1[word1[j]] += 1
        
        return valid_count