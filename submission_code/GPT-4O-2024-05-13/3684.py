class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern p into two parts around the '*'
        parts = p.split('*')
        prefix, suffix = parts[0], parts[1]
        
        # Check if the string s contains the prefix and suffix in the correct order
        if prefix in s:
            # Find the starting index of the prefix in s
            start_index = s.find(prefix)
            # Check if the remaining part of s after the prefix contains the suffix
            return suffix in s[start_index + len(prefix):]
        
        return False