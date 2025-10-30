from collections import defaultdict

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        m = len(word2)
        n = len(word1)
        if m > n:
            return 0
        
        required = defaultdict(int)
        for c in word2:
            required[c] += 1
        required_size = len(required)
        
        left = 0
        formed = 0
        count = defaultdict(int)
        total = 0
        
        for right in range(n):
            c = word1[right]
            if c in required:
                count[c] += 1
                if count[c] == required[c]:
                    formed += 1
            
            # Try to contract the window as much as possible
            while formed == required_size:
                # Calculate contribution if the current window is valid and long enough
                if right >= m - 1:
                    a = right - m + 1
                    b = min(left, a)
                    contribution = b + 1
                    total += contribution
                
                # Move the left pointer to try to reduce the window
                left_char = word1[left]
                if left_char in required:
                    count[left_char] -= 1
                    if count[left_char] < required[left_char]:
                        formed -= 1
                left += 1
        
        return total