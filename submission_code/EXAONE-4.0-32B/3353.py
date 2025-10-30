class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        n = len(s)
        if n < 2:
            return False
        
        rev = s[::-1]
        rev_subs = set()
        for i in range(len(rev) - 1):
            rev_subs.add(rev[i:i+2])
        
        for i in range(len(s) - 1):
            substr = s[i:i+2]
            if substr in rev_subs:
                return True
        
        return False