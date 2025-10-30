class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n-2):
            if nums[i] < nums[i+1] + nums[i+2]:
                ans = max(ans, nums[i] * nums[i+1] * nums[i+2])
        return ans