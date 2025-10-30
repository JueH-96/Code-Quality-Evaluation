from collections import Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        len_word2 = len(word2)
        counter_word2 = Counter(word2)
        total = 0
        current_counter = Counter()
        left = 0
        
        for right in range(len(word1)):
            current_counter[word1[right]] += 1
            while right - left + 1 > len_word2:
                current_counter[word1[left]] -= 1
                if current_counter[word1[left]] == 0:
                    del current_counter[word1[left]]
                left += 1
            if right - left + 1 == len_word2:
                valid = True
                for char in counter_word2:
                    if current_counter.get(char, 0) < counter_word2[char]:
                        valid = False
                        break
                if valid:
                    total += 1
        return total