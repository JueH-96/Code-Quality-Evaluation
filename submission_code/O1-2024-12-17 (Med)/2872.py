class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        res = 0
        merged_sum = nums[-1]
        
        # Traverse from right to left
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= merged_sum:
                # We can merge the current element into the ongoing sum
                merged_sum += nums[i]
            else:
                # Finalize the current sum and start a new one
                res = max(res, merged_sum)
                merged_sum = nums[i]
        
        # Check final sum segment
        res = max(res, merged_sum)
        return res