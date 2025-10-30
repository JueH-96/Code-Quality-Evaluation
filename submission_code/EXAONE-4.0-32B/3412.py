class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        index_map = {}
        for idx, char in enumerate(s):
            index_map[char] = idx
        
        total_diff = 0
        for i, char in enumerate(t):
            original_index = index_map[char]
            total_diff += abs(original_index - i)
        
        return total_diff