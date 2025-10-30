class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        count = 0
        words_unpaired = set(words)
        for word in words:
            if word in words_unpaired:
                rev = word[::-1]
                if rev in words_unpaired and rev != word:
                    count += 1
                    words_unpaired.remove(word)
                    words_unpaired.remove(rev)
        return count