class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        if target > total_sum * (target // total_sum + 2) :
            return -1
            
        target %= total_sum
        if target == 0:
            min_len = float('inf')
            curr_sum = 0
            left = 0
            for right in range(2 * n):
                curr_sum += nums[right % n]
                while curr_sum >= total_sum:
                    min_len = min(min_len, right - left + 1)
                    curr_sum -= nums[left % n]
                    left += 1
            if min_len == float('inf'):
                return -1
            return min_len
        
        
        min_len = float('inf')
        curr_sum = 0
        left = 0
        for right in range(2 * n):
            curr_sum += nums[right % n]
            while curr_sum >= target:
                min_len = min(min_len, right - left + 1)
                curr_sum -= nums[left % n]
                left += 1
        if min_len == float('inf'):
            return -1
        return min_len