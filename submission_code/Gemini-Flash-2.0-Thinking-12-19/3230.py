class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        def is_almost_equal(c1, c2):
            return c1 == c2 or abs(ord(c1) - ord(c2)) == 1

        word_list = list(word)
        count = 0
        for i in range(len(word_list) - 1):
            if is_almost_equal(word_list[i], word_list[i+1]):
                count += 1
                if word_list[i] <= 'b':
                    word_list[i+1] = 'z'
                else:
                    word_list[i+1] = 'a'
        return count