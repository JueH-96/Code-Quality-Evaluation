class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        operations = 0
        i = 0
        while i < n - 1:
            current = word[i]
            next_char = word[i+1]
            if current == next_char or abs(ord(current) - ord(next_char)) == 1:
                operations += 1
                i += 2  # Skip the next character since we changed the current one
            else:
                i += 1
        return operations