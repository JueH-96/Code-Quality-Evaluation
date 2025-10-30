class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        # Iterate over the string with a 1-indexed position
        for index, char in enumerate(s, start=1):
            # Calculate the reversed alphabet position: 'a' -> 26, 'b' -> 25, etc.
            reversed_position = 27 - (ord(char) - ord('a') + 1)
            total += reversed_position * index
        return total