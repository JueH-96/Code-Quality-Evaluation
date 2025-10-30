class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        current = []

        for char in target:
            # Key 1: append 'a'
            current.append('a')
            res.append("".join(current))

            # Key 2: increment until last character matches char
            # Calculate how many increments needed from 'a' to char
            needed = ord(char) - ord('a')
            for _ in range(needed):
                # If it's 'z', wrap to 'a', otherwise just go to next character
                if current[-1] == 'z':
                    current[-1] = 'a'
                else:
                    current[-1] = chr(ord(current[-1]) + 1)
                res.append("".join(current))

        return res