class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern around the '*'
        left, right = p.split('*')
        
        # Check for possible start and end positions of the pattern in the string
        for start in range(len(s) - len(left) + 1):
            if s[start:start+len(left)] == left:
                # Check if the right part of the pattern can match after the left part
                end = start + len(left)
                if s[end:end+len(right)] == right and end + len(right) <= len(s):
                    return True
        return False