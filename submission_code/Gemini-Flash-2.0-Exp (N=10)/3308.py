class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        if not s:
            return ""
        
        history = []
        while s:
            history.append(s)
            new_s = ""
            seen = set()
            for char in s:
                if char not in seen:
                    seen.add(char)
                else:
                    new_s += char
            s = new_s
        
        if len(history) > 1:
            return history[-2]
        else:
            return history[0]