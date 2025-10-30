class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        counts = {}
        for num in nums:
            for i in range(max(0, num - k), min(100001, num + k + 1)):
                counts[i] = counts.get(i, 0) + 1
        return max(counts.values())