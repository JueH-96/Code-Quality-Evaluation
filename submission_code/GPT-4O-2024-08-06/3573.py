from collections import Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # Create a counter for word2
        word2_counter = Counter(word2)
        word2_length = len(word2)
        
        # Initialize a counter for the current window in word1
        current_counter = Counter()
        
        # Initialize the number of valid substrings
        valid_count = 0
        
        # Use a sliding window approach
        left = 0
        for right in range(len(word1)):
            # Add the current character to the current window counter
            current_counter[word1[right]] += 1
            
            # If the window size exceeds word2_length, shrink it from the left
            if right - left + 1 > word2_length:
                current_counter[word1[left]] -= 1
                if current_counter[word1[left]] == 0:
                    del current_counter[word1[left]]
                left += 1
            
            # Check if the current window can be rearranged to have word2 as a prefix
            if right - left + 1 == word2_length:
                if all(current_counter[char] >= word2_counter[char] for char in word2_counter):
                    valid_count += 1
        
        return valid_count