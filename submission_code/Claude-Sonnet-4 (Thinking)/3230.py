class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        def are_almost_equal(a, b):
            return a == b or abs(ord(a) - ord(b)) == 1
        
        word = list(word)
        operations = 0
        
        for i in range(1, len(word)):
            if are_almost_equal(word[i], word[i-1]):
                operations += 1
                # Find a character that is not almost-equal to word[i-1] and word[i+1] (if exists)
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if not are_almost_equal(c, word[i-1]) and (i+1 >= len(word) or not are_almost_equal(c, word[i+1])):
                        word[i] = c
                        break
        
        return operations