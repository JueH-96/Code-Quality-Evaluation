class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_val = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                for k in range(j+1, len(nums)):
                    val = (nums[i] - nums[j]) * nums[k]
                    if val > max_val:
                        max_val = val
        return max_val