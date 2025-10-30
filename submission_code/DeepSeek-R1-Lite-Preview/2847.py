class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        word_set = set(words)
        counter = 0
        for word in words:
            if word in word_set:
                reversed_word = word[::-1]
                if reversed_word in word_set and reversed_word != word:
                    counter += 1
                    word_set.remove(word)
                    word_set.remove(reversed_word)
                else:
                    word_set.remove(word)
        return counter