class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        res = 0
        low, high = 1, n
        
        while low <= high:
            mid = (low + high) // 2
            found = False
            for l in range(n - mid + 1):
                r = l + mid - 1
                m = l + (mid - 1) // 2  # Median index
                # Calculate sum of absolute differences to the median
                sum_left = nums[m] * (m - l + 1) - (prefix[m + 1] - prefix[l])
                sum_right = (prefix[r + 1] - prefix[m + 1]) - nums[m] * (r - m)
                total_cost = sum_left + sum_right
                if total_cost <= k:
                    found = True
                    break
            if found:
                res = mid
                low = mid + 1
            else:
                high = mid - 1
        return res