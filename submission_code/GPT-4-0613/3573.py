class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        word2_counter = Counter(word2)
        word1_counter = Counter()
        left = 0
        count = 0
        for right in range(len(word1)):
            word1_counter[word1[right]] += 1
            while all(word1_counter[char] >= word2_counter[char] for char in word2_counter):
                count += len(word1) - right
                word1_counter[word1[left]] -= 1
                left += 1
        return count