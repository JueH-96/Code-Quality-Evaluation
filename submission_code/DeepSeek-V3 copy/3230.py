class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        operations = 0
        i = 0
        while i < n - 1:
            a = word[i]
            b = word[i+1]
            if a == b or abs(ord(a) - ord(b)) == 1:
                operations += 1
                i += 2  # Skip the next character since we changed the current one
            else:
                i += 1
        return operations