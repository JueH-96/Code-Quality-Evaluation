from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        max_num = 0
        if not nums:
            return False
        for num in nums:
            max_num = max(max_num, num)
        n = max_num
        if len(nums) != n + 1:
            return False
        base_counts = Counter()
        for i in range(1, n):
            base_counts[i] += 1
        base_counts[n] += 2
        nums_counts = Counter(nums)
        return nums_counts == base_counts