class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set()
        pairs = 0
        for word in words:
            rev_word = word[::-1]
            if rev_word in seen:
                pairs += 1
            seen.add(word)
        return pairs