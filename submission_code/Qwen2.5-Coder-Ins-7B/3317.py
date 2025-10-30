class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        total_chars = sum(len(word) for word in words)
        odd_count = sum(word.count(char) % 2 for word in words for char in set(word))
        pairs = total_chars // 2
        odd_chars = odd_count // 2
        return min(pairs + odd_chars, len(words))