class Solution:
    def findLatestTime(self, s: str) -> str:
        s = list(s)
        
        if s[0] == "?":
            s[0] = "1" if s[1] in "012345678?" else "0"
        
        if s[1] == "?":
            s[1] = "9" if s[0] == "0" else "1"
        
        if s[3] == "?":
            s[3] = "5"
        
        if s[4] == "?":
            s[4] = "9"
        
        return "".join(s)