class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        word_list = list(word)
        operations = 0
        for i in range(1, len(word_list)):
            c1 = word_list[i-1]
            c2 = word_list[i]
            if c1 == c2 or abs(ord(c1) - ord(c2)) == 1:
                operations += 1
                word_list[i] = 'c'
        return operations