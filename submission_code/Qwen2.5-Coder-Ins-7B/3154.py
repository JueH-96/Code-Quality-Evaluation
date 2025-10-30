class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_val = 0
        n = len(nums)
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    max_val = max(max_val, (nums[i] - nums[j]) * nums[k])
        return max_val