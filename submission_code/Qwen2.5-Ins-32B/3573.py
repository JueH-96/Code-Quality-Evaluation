from collections import Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        target_count = Counter(word2)
        target_len = len(word2)
        result = 0
        left = 0
        current_count = Counter()
        
        for right in range(len(word1)):
            current_count[word1[right]] += 1
            
            while current_count[word1[right]] > target_count[word1[right]]:
                current_count[word1[left]] -= 1
                left += 1
            
            if right - left + 1 >= target_len:
                valid = True
                for char in target_count:
                    if current_count[char] < target_count[char]:
                        valid = False
                        break
                if valid:
                    result += left + 1
        
        return result