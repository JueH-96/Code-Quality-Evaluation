class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp_even = [0] * n
        dp_odd = [0] * n
        parity = nums[0] % 2
        
        if parity == 0:
            dp_even[0] = nums[0]
            dp_odd[0] = nums[0] - x
        else:
            dp_even[0] = nums[0] - x
            dp_odd[0] = nums[0]
        
        for i in range(1, n):
            if nums[i] % 2 == 0:
                dp_even[i] = max(dp_even[i-1] + nums[i], dp_odd[i-1] + nums[i] - x)
                dp_odd[i] = dp_odd[i-1] + nums[i]
            else:
                dp_even[i] = dp_even[i-1] + nums[i]
                dp_odd[i] = max(dp_odd[i-1] + nums[i], dp_even[i-1] + nums[i] - x)
        
        return max(dp_even[-1], dp_odd[-1])