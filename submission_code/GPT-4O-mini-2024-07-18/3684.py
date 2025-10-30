class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern into two parts around the '*'
        prefix, suffix = p.split('*')
        
        # Check if the prefix can be found at the start of s
        if not s.startswith(prefix):
            return False
        
        # Check if the suffix can be found at the end of s
        if not s.endswith(suffix):
            return False
        
        # If both prefix and suffix are matched, we need to ensure
        # that the length of s is at least the combined length of prefix and suffix
        if len(s) < len(prefix) + len(suffix):
            return False
        
        return True