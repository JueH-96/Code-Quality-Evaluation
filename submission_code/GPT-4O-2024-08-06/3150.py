class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_length = float('inf')
        result = ""
        
        # Sliding window approach
        left = 0
        count_ones = 0
        
        for right in range(n):
            if s[right] == '1':
                count_ones += 1
            
            # When we have exactly k ones, try to shrink the window
            while count_ones > k:
                if s[left] == '1':
                    count_ones -= 1
                left += 1
            
            # Check if we have exactly k ones
            if count_ones == k:
                current_length = right - left + 1
                if current_length < min_length:
                    min_length = current_length
                    result = s[left:right+1]
                elif current_length == min_length:
                    # Compare lexicographically
                    candidate = s[left:right+1]
                    if candidate < result:
                        result = candidate
        
        return result