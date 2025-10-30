class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        word_set = set(words)
        count = 0
        for w in words:
            if w in word_set:
                rev = w[::-1]
                if rev in word_set and rev != w:  # they must be distinct
                    count += 1
                    word_set.remove(w)
                    word_set.remove(rev)
        return count