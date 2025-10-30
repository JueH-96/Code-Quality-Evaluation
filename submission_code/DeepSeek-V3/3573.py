from collections import defaultdict

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # Count the frequency of each character in word2
        word2_counts = defaultdict(int)
        for char in word2:
            word2_counts[char] += 1
        
        # Total characters needed from word2
        total_needed = len(word2)
        
        # Initialize the count of valid substrings
        valid_count = 0
        
        # Initialize the frequency map for the current window in word1
        window_counts = defaultdict(int)
        left = 0
        # The window size must be at least the length of word2
        # So we start with a window of size len(word2)
        for right in range(len(word1)):
            char = word1[right]
            window_counts[char] += 1
            
            # If the window size is greater than or equal to len(word2)
            if right - left + 1 >= total_needed:
                # Check if the current window can form word2 as a prefix
                # We need to have at least the required counts for each character in word2
                valid = True
                for key in word2_counts:
                    if window_counts[key] < word2_counts[key]:
                        valid = False
                        break
                if valid:
                    valid_count += 1
                
                # Move the left pointer to the right
                left_char = word1[left]
                window_counts[left_char] -= 1
                if window_counts[left_char] == 0:
                    del window_counts[left_char]
                left += 1
        
        return valid_count