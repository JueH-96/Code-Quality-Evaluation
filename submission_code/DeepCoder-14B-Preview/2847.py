class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        words_set = set(words)
        used = set()
        count = 0
        for word in words:
            if word not in used:
                rev_word = word[::-1]
                if rev_word in words_set and rev_word != word:
                    if rev_word not in used:
                        used.add(word)
                        used.add(rev_word)
                        count += 1
        return count