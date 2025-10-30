class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""

        for char in target:
            while current and (ord(char) < ord(current[-1]) or (ord(char) - ord(current[-1]) > 1)):
                current = current[:-1]
            for _ in range(ord(char) - ord(current[-1]) if current else ord(char) - ord('a') + 1):
                current += 'a'
                result.append(current)
            current = current[:-1] + char
            result.append(current)

        return result