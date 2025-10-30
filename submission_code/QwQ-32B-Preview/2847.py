class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set()
        pair_count = 0
        for word in words:
            rev = word[::-1]
            if rev in seen:
                pair_count += 1
            else:
                seen.add(word)
        return pair_count