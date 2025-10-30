class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        word_set = set(words)
        pairs = 0

        for w in words:
            # If w was already used in a pair, skip it
            if w not in word_set:
                continue

            rev = w[::-1]
            # We cannot form a pair if reversed is the same (palindrome) because words are distinct
            if rev in word_set and rev != w:
                pairs += 1
                # Remove both strings so they aren't used again
                word_set.remove(w)
                word_set.remove(rev)

        return pairs