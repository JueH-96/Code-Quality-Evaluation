class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        last = -float('inf')
        count = 0
        for num in nums:
            low = num - k
            high = num + k
            candidate = max(low, last + 1)
            if candidate <= high:
                count += 1
                last = candidate
        return count