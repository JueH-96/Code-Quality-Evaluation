class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        def are_almost_equal(a, b):
            return a == b or abs(ord(a) - ord(b)) == 1

        operations = 0
        i = 1
        while i < len(word):
            if are_almost_equal(word[i-1], word[i]):
                operations += 1
                i += 2
            else:
                i += 1
        return operations