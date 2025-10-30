class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        from collections import Counter
        counter = Counter()
        for word in words:
            counter += Counter(word)
        odd = sum(v % 2 for v in counter.values())
        return len(words) - max(0, odd - 1)