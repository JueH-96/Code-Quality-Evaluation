class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        index_map = {char: i for i, char in enumerate(s)}
        return sum(abs(index_map[char] - i) for i, char in enumerate(t))