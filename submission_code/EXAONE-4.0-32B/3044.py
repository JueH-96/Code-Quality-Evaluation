class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected = set()
        n = len(nums)
        for i in range(n-1, -1, -1):
            if nums[i] <= k:
                if nums[i] not in collected:
                    collected.add(nums[i])
            if len(collected) == k:
                return n - i
        return n