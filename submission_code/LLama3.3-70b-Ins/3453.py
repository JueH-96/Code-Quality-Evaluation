from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        def is_valid(s: str) -> bool:
            """Check if a string is valid."""
            for i in range(len(s) - 1):
                if s[i:i+2] == "00":
                    return False
            return True

        def generate_strings(n: int) -> List[str]:
            """Generate all valid strings of length n."""
            if n == 1:
                return ["0", "1"]
            else:
                prev_strings = generate_strings(n - 1)
                valid_strings = []
                for s in prev_strings:
                    valid_strings.append("0" + s)
                    valid_strings.append("1" + s)
                return [s for s in valid_strings if is_valid(s)]

        return generate_strings(n)