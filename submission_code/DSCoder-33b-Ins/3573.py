class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        count1 = Counter(word2)
        count2 = Counter(word1[:len(word2)])
        valid_count = 0
        for i in range(len(word2), len(word1)+1):
            if count1 == count2:
                valid_count += 1
            if i < len(word1):
                count2[word1[i]] += 1
                if count2[word1[i-len(word2)]] == 1:
                    del count2[word1[i-len(word2)]]
                else:
                    count2[word1[i-len(word2)]] -= 1
        return valid_count