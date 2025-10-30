class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        low = 1
        high = n
        ans = 1
        
        while low <= high:
            mid = (low + high) // 2
            found = False
            for l in range(n - mid + 1):
                r = l + mid - 1
                m = (l + r) // 2
                left_count = m - l + 1
                left_sum = prefix_sum[m + 1] - prefix_sum[l]
                right_sum = prefix_sum[r + 1] - prefix_sum[m + 1]
                sum_abs = (nums[m] * left_count - left_sum) + (right_sum - nums[m] * (r - m))
                if sum_abs <= k:
                    found = True
                    break
            if found:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return ans