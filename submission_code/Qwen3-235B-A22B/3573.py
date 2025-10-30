from collections import defaultdict, Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        required_count = Counter(word2)
        required_chars = len(required_count)
        window_counts = defaultdict(int)
        count_chars = 0
        left = 0
        total = 0
        L = len(word2)
        
        for right in range(len(word1)):
            char = word1[right]
            if char in required_count:
                window_counts[char] += 1
                if window_counts[char] == required_count[char]:
                    count_chars += 1
            
            # Shrink the window as much as possible while valid
            while count_chars == required_chars:
                current_left = left
                char_left = word1[left]
                if char_left in required_count:
                    window_counts[char_left] -= 1
                    if window_counts[char_left] < required_count[char_left]:
                        count_chars -= 1
                left += 1
            
            current_left = left - 1
            if current_left >= 0:
                possible_max_start = right - L + 1
                if possible_max_start >= 0:
                    min_val = min(current_left, possible_max_start)
                    total += (min_val + 1)
        
        return total