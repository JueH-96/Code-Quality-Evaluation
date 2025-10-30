class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        def are_almost_equal(a, b):
            return abs(ord(a) - ord(b)) <= 1

        word = list(word)
        n = len(word)
        operations = 0

        for i in range(1, n):
            if are_almost_equal(word[i], word[i-1]):
                # Change word[i] to a character that is not almost-equal to word[i-1]
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if not are_almost_equal(c, word[i-1]):
                        word[i] = c
                        operations += 1
                        break

        return operations