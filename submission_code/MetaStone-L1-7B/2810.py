class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_total = float('inf')
        for k in range(n):
            current_sum = 0
            for t in range(n):
                pos = (t - k) % n
                current_sum += nums[pos]
            total = current_sum + k * x
            if total < min_total:
                min_total = total
        return min_total