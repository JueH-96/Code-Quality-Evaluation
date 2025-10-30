from collections import defaultdict

class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        count = defaultdict(int)
        n = len(s)
        left = 0
        countOverK = 0
        valid_count = 0  # Number of invalid substrings (valid in our sliding window)
        
        for right in range(n):
            current_char = s[right]
            count[current_char] += 1
            if count[current_char] == k:
                countOverK += 1
            
            # Shrink the window until no character has count >= k
            while countOverK > 0:
                left_char = s[left]
                prev_count = count[left_char]
                count[left_char] -= 1
                if prev_count == k:
                    countOverK -= 1
                left += 1
            
            # All substrings ending at 'right' with start >= 'left' are invalid
            valid_count += right - left + 1
        
        total_substrings = n * (n + 1) // 2
        return total_substrings - valid_count