class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Split the pattern into three parts based on the '*' characters
        parts = p.split('*')
        
        # Extract the prefix and suffix from the pattern
        prefix = parts[0]
        suffix = parts[2]
        
        # Find the starting index for the prefix in s
        start_index = 0
        while start_index <= len(s) - len(prefix):
            if s[start_index:start_index + len(prefix)] == prefix:
                break
            start_index += 1
        else:
            return -1  # No valid starting point for prefix
        
        # Find the ending index for the suffix in s
        end_index = len(s) - len(suffix)
        while end_index >= 0:
            if s[end_index:end_index + len(suffix)] == suffix:
                break
            end_index -= 1
        else:
            return -1  # No valid ending point for suffix
        
        # Now we need to find the shortest substring that matches
        # We need to ensure that the start_index is before end_index
        min_length = float('inf')
        
        for i in range(start_index, len(s)):
            if s[i:i + len(prefix)] == prefix:
                for j in range(end_index, -1, -1):
                    if s[j:j + len(suffix)] == suffix and j >= i + len(prefix):
                        min_length = min(min_length, j + len(suffix) - i)
        
        return min_length if min_length != float('inf') else -1