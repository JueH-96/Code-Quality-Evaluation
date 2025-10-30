class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        if k > n:
            return False
            
        for i in range(n - k + 1):
            substring = s[i:i+k]
            # Check if the substring has exactly one distinct character
            if substring.count(substring[0]) != k:
                continue  # Not all same, skip to next
            
            char = substring[0]
            
            # Check the preceding character (if exists)
            if i > 0 and s[i - 1] == char:
                continue
            
            # Check the succeeding character (if exists)
            if i + k < n and s[i + k] == char:
                continue
            
            # If all conditions are met, return True immediately
            return True
            
        # If no substring met the criteria, return False
        return False