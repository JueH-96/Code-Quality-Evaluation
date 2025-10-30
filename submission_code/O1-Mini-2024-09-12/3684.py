class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern into prefix and suffix around the '*'
        prefix, suffix = p.split('*')
        len_p = len(prefix) + len(suffix)
        
        # Iterate through all possible substrings of s with length >= len_p
        for i in range(len(s) - len_p + 1):
            # Check if the substring starts with the prefix
            if s.startswith(prefix, i):
                # Calculate the start position for the suffix
                start_suffix = i + len(prefix)
                # Iterate through possible end positions for the prefix and suffix
                for j in range(start_suffix, len(s) + 1):
                    # Check if the substring ends with the suffix
                    if s.startswith(suffix, j - len(suffix)) and j - i >= len_p:
                        return True
        return False