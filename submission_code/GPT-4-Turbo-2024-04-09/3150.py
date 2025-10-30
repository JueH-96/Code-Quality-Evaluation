class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        best_substring = ""
        
        # Two pointers technique to find the shortest substring with exactly k '1's
        left = 0
        count_ones = 0
        
        for right in range(n):
            if s[right] == '1':
                count_ones += 1
            
            # When we have exactly k '1's, try to minimize the window
            while count_ones > k:
                if s[left] == '1':
                    count_ones -= 1
                left += 1
            
            # Check if we have a valid window
            if count_ones == k:
                # Adjust left pointer to make sure we have the smallest window
                while s[left] == '0':
                    left += 1
                
                # Calculate the length of the current valid window
                current_length = right - left + 1
                if current_length < min_len:
                    min_len = current_length
                    best_substring = s[left:right+1]
                elif current_length == min_len:
                    # Check lexicographical order if lengths are the same
                    current_substring = s[left:right+1]
                    if current_substring < best_substring:
                        best_substring = current_substring
        
        return best_substring