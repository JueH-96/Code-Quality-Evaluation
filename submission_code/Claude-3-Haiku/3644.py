class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = float('inf')
        
        for i in range(n):
            curr_sum = 0
            for j in range(i, n):
                curr_sum += nums[j]
                if l <= j - i + 1 <= r and curr_sum > 0 and curr_sum < min_sum:
                    min_sum = curr_sum
        
        return min_sum if min_sum != float('inf') else -1