class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        word_set = set(words)
        pairs = 0
        for word in words:
            if word[::-1] in word_set:
                pairs += 1
                word_set.remove(word)
                word_set.remove(word[::-1])
        return pairs