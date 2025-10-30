class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        res = ""
        while True:
            temp = ""
            removed = False
            seen = set()
            for char in s:
                if 'a' <= char <= 'z' and char not in seen:
                    seen.add(char)
                    removed = True
                else:
                    temp += char
            if not removed:
                return s
            res = s
            s = temp