class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        while s:
            previous_s = s
            removal_indices = set()
            for char_code in range(ord('a'), ord('z') + 1):
                char = chr(char_code)
                try:
                    first_occurrence = s.index(char)
                    removal_indices.add(first_occurrence)
                except ValueError:
                    pass

            new_s = "".join([char for i, char in enumerate(s) if i not in removal_indices])
            s = new_s

            if not s:
                return previous_s
        return ""