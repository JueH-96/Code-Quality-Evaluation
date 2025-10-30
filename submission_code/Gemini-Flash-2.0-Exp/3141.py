class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        if target > total_sum and total_sum > 0:
            k = target // total_sum
            target %= total_sum
            ans = k * n
        else:
            ans = 0
        
        min_len = float('inf')
        curr_sum = 0
        left = 0
        
        for right in range(2 * n):
            curr_sum += nums[right % n]
            
            while curr_sum > target:
                curr_sum -= nums[left % n]
                left += 1
            
            if curr_sum == target:
                min_len = min(min_len, right - left + 1)
        
        if min_len == float('inf'):
            if ans > 0:
                return -1
            else:
                return -1
        else:
            return ans + min_len