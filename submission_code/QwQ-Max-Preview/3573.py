from collections import defaultdict

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # Calculate the required character counts from word2
        required = defaultdict(int)
        for c in word2:
            required[c] += 1
        required_chars = set(required.keys())
        
        # Check if word1 has enough characters to meet word2's requirements
        word1_counts = defaultdict(int)
        for c in word1:
            word1_counts[c] += 1
        for c in required_chars:
            if word1_counts[c] < required[c]:
                return 0
        
        n = len(word1)
        k = len(word2)
        if k > n:
            return 0
        
        current_freq = defaultdict(int)
        left = 0
        right = 0
        count = 0
        
        while left < n:
            # Expand the window until the current_freq meets required and the window length is at least k
            while (right < n) and (right - left < k - 1 or not all(current_freq[c] >= required[c] for c in required_chars)):
                current_freq[word1[right]] += 1
                right += 1
            
            # Check if the current window meets the requirements
            if all(current_freq[c] >= required[c] for c in required_chars) and (right - left >= k - 1):
                # All substrings starting at left and ending from right-1 to n-1 are valid
                count += (n - right + 1)
            else:
                # No more valid windows possible
                break
            
            # Move the left pointer to the next position
            current_freq[word1[left]] -= 1
            left += 1
        
        return count