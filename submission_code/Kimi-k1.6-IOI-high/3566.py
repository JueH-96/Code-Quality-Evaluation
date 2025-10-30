class Solution:
    def stringSequence(self, target: str) -> List[str]:
        current = []
        result = []
        for c in target:
            current.append('a')
            result.append(''.join(current))
            needed = ord(c) - ord('a')
            for _ in range(needed):
                # Increment the last character
                last = current[-1]
                new_last = chr((ord(last) - ord('a') + 1) % 26 + ord('a'))
                current[-1] = new_last
                result.append(''.join(current))
        return result