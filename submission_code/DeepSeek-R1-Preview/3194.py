class Solution:
    def findWordsContaining(self, words, x):
        result = []
        for index, word in enumerate(words):
            if x in word:
                result.append(index)
        return result