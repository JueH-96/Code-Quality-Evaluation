class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set()
        count = 0
        for word in words:
            rev = word[::-1]
            if rev in seen:
                count += 1
                seen.remove(rev)
            else:
                seen.add(word)
        return count