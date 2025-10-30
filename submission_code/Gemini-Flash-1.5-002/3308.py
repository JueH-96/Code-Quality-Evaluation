class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        removed = set()
        last_non_empty = ""
        
        while len(s) > 0:
            last_non_empty = s
            new_s = ""
            removed_chars = set()

            for char in s:
                if char not in removed:
                    removed.add(char)
                    removed_chars.add(char)
                else:
                    new_s += char
            s = new_s

        return last_non_empty