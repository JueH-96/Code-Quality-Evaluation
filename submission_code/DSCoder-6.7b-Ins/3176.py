class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return -1
        nums.sort()
        res = float('inf')
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    if nums[i] < nums[j] < nums[k] and nums[i] + nums[j] + nums[k] < res:
                        res = nums[i] + nums[j] + nums[k]
        return res if res != float('inf') else -1