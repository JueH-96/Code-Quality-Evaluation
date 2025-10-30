class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import defaultdict
        
        target_count = defaultdict(int)
        for char in word2:
            target_count[char] += 1
        
        need = len(target_count)
        have = defaultdict(int)
        required = {char: 0 for char in target_count}
        for char in target_count:
            required[char] = target_count[char]
        
        left = 0
        total = 0
        formed = 0
        
        for right, char in enumerate(word1):
            if char in required:
                have[char] += 1
                if have[char] >= required[char]:
                    formed += 1
            
            while formed == need and left <= right:
                window_length = right - left + 1
                if window_length >= len(word2):
                    total += len(word1) - right
                if word1[left] in required:
                    have[word1[left]] -= 1
                    if have[word1[left]] < required[word1[left]]:
                        formed -= 1
                left += 1
        
        return total