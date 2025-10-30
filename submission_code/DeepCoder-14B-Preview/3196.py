class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        nums.sort()
        pre_sum = [0] * (n + 1)
        for i in range(n):
            pre_sum[i + 1] = pre_sum[i] + nums[i]
        
        def is_possible(f):
            if f == 0:
                return False
            for l in range(n - f + 1):
                r = l + f - 1
                m = l + (f - 1) // 2
                sum_left = pre_sum[m + 1] - pre_sum[l]
                sum_right = pre_sum[r + 1] - pre_sum[m + 1]
                
                left_cost = nums[m] * (m - l + 1) - sum_left
                right_cost = sum_right - nums[m] * (r - m)
                
                if left_cost + right_cost <= k:
                    return True
            return False
        
        low = 1
        high = n
        best = 0
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best