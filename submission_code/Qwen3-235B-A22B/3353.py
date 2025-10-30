class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        if len(s) < 2:
            return False
        
        reversed_s = s[::-1]
        rev_subs = set()
        
        for i in range(len(reversed_s) - 1):
            rev_subs.add(reversed_s[i:i+2])
        
        for i in range(len(s) - 1):
            if s[i:i+2] in rev_subs:
                return True
        
        return False