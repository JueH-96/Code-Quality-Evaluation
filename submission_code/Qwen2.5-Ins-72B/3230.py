class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        def is_almost_equal(a, b):
            return abs(ord(a) - ord(b)) <= 1
        
        operations = 0
        i = 1
        while i < len(word):
            if is_almost_equal(word[i], word[i - 1]):
                operations += 1
                i += 2
            else:
                i += 1
        return operations