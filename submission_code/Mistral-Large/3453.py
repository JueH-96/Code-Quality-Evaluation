from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        def is_valid(s: str) -> bool:
            for i in range(len(s) - 1):
                if s[i] == '0' and s[i + 1] == '0':
                    return False
            return True

        def generate_strings(n: int) -> List[str]:
            if n == 0:
                return [""]
            smaller_strings = generate_strings(n - 1)
            valid_strings = []
            for s in smaller_strings:
                if is_valid("0" + s):
                    valid_strings.append("0" + s)
                if is_valid("1" + s):
                    valid_strings.append("1" + s)
            return valid_strings

        return generate_strings(n)