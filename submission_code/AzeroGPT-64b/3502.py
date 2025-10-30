from collections import Counter

class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        size = len(s)
        window_start = 0
        resulting_counter = 0
        char_frequencies = Counter()
        valid_characters = 0
        
        for window_end in range(size):
            right_char = s[window_end]
            char_frequencies[right_char] += 1
            
            if char_frequencies[right_char] == k:
                valid_characters += 1
            
            while valid_characters == len(char_frequencies):
                resulting_counter += (size - window_end)
                left_char = s[window_start]
                char_frequencies[left_char] -= 1
                if char_frequencies[left_char] == k - 1:
                    valid_characters -= 1
                window_start += 1
        return resulting_counter