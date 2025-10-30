class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern p into two parts around the '*'
        parts = p.split('*')
        prefix = parts[0]
        suffix = parts[1]
        
        # Check if the string s starts with the prefix and ends with the suffix
        # The substring between prefix and suffix can be any sequence including empty
        for i in range(len(s) - len(prefix) - len(suffix) + 1):
            if s.startswith(prefix, i) and s.endswith(suffix, i + len(prefix)):
                return True
        return False