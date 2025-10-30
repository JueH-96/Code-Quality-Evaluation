class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        rev_words = [word[::-1] for word in words]
        return sum(word in rev_words[i+1:] for i, word in enumerate(words))