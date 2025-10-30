from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        word_set = set(words)
        used = set()
        count = 0
        for word in words:
            if word not in used:
                rev = word[::-1]
                if rev in word_set and rev not in used and rev != word:
                    used.add(word)
                    used.add(rev)
                    count += 1
        return count