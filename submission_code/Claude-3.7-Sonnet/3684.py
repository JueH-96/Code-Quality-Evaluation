class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern into parts before and after the '*'
        parts = p.split('*')
        before_star = parts[0]
        after_star = parts[1]
        
        # If both before_star and after_star are empty, return True
        if not before_star and not after_star:
            return True
        
        # If before_star is empty, check if after_star is a substring of s
        if not before_star:
            return after_star in s
        
        # If after_star is empty, check if before_star is a substring of s
        if not after_star:
            return before_star in s
        
        # Check for each position in s
        for i in range(len(s) - len(before_star) + 1):
            if s[i:i+len(before_star)] == before_star:
                for j in range(i + len(before_star), len(s) - len(after_star) + 1):
                    if s[j:j+len(after_star)] == after_star:
                        return True
        
        return False