from collections import Counter

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        freq = Counter(nums)
        for f, t in zip(moveFrom, moveTo):
            count = freq.pop(f)
            freq[t] = freq.get(t, 0) + count
        return sorted(freq.keys())