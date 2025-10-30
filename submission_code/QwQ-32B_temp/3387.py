class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        m = n // 2
        total = 0
        
        for i in range(n):
            if i < m:
                if nums[i] > k:
                    total += nums[i] - k
            elif i == m:
                total += abs(nums[i] - k)
            else:
                if nums[i] < k:
                    total += k - nums[i]
        
        return total