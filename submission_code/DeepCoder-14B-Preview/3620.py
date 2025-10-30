class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums_sorted = sorted(nums)
        prev = float('-inf')
        count = 0
        for num in nums_sorted:
            lower = num - k
            upper = num + k
            new_val = max(prev + 1, lower)
            if new_val > upper:
                continue
            prev = new_val
            count += 1
        return count