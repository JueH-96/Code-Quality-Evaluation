class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp_even = [-float('inf')] * n
        dp_odd = [-float('inf')] * n
        if nums[0] % 2 == 0:
            dp_even[0] = nums[0]
        else:
            dp_odd[0] = nums[0]
        
        for i in range(1, n):
            if nums[i] % 2 == 0:
                max_prev_even = -float('inf')
                max_prev_odd_minus_x = -float('inf')
                for j in range(i):
                    if nums[j] % 2 == 0:
                        max_prev_even = max(max_prev_even, dp_even[j])
                    else:
                        max_prev_odd_minus_x = max(max_prev_odd_minus_x, dp_odd[j] - x)
                dp_even[i] = nums[i] + max(max_prev_even, max_prev_odd_minus_x, -float('inf'))
            else:
                max_prev_odd = -float('inf')
                max_prev_even_minus_x = -float('inf')
                for j in range(i):
                    if nums[j] % 2 != 0:
                        max_prev_odd = max(max_prev_odd, dp_odd[j])
                    else:
                        max_prev_even_minus_x = max(max_prev_even_minus_x, dp_even[j] - x)
                dp_odd[i] = nums[i] + max(max_prev_odd, max_prev_even_minus_x, -float('inf'))
                
        return max(max(dp_even), max(dp_odd))