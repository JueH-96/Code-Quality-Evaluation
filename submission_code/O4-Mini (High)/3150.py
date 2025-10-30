class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        # Collect the indices of all '1's in the string
        positions = [i for i, ch in enumerate(s) if ch == '1']
        # If there are fewer than k '1's, no beautiful substring exists
        if len(positions) < k:
            return ""
        
        min_len = float('inf')
        result = ""
        # Slide a window of size k over the positions array
        for i in range(len(positions) - k + 1):
            left = positions[i]
            right = positions[i + k - 1]
            curr_len = right - left + 1
            substr = s[left:right + 1]
            # Update if we find a shorter substring or a lexicographically smaller one
            if curr_len < min_len or (curr_len == min_len and substr < result):
                min_len = curr_len
                result = substr
        
        return result