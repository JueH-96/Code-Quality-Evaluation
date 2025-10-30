class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        if n < k:
            return ""
        
        # Count the number of 1's in the string
        count = s.count('1')
        if count < k:
            return ""
        
        # Initialize the minimum length and the result string
        min_len = float('inf')
        result = ""
        
        # Iterate through the string
        for i in range(n - k + 1):
            # Count the number of 1's in the current substring
            curr_count = s[i:i+k].count('1')
            
            # If the current substring has exactly k 1's, update the minimum length and result
            if curr_count == k:
                if i + k <= n and s[i:i+k] < result or not result:
                    min_len = k
                    result = s[i:i+k]
        
        return result