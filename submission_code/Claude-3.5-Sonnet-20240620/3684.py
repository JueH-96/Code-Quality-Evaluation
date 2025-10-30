class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern into two parts: before and after the '*'
        before, after = p.split('*')
        
        # Check if the pattern starts with the 'before' part
        if not s.startswith(before):
            return False
        
        # Check if the pattern ends with the 'after' part
        if not s.endswith(after):
            return False
        
        # Calculate the minimum length required for a match
        min_length = len(before) + len(after)
        
        # If the string is shorter than the minimum length, it can't match
        if len(s) < min_length:
            return False
        
        # If there's no 'after' part, we've already confirmed the match
        if not after:
            return True
        
        # Find all possible positions where 'after' could start
        start_pos = len(before)
        while True:
            pos = s.find(after, start_pos)
            if pos == -1:
                return False
            if pos + len(after) == len(s):
                return True
            start_pos = pos + 1