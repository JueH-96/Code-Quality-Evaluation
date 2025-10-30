class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        def is_possible(m):
            required = m
            for i in range(n - m + 1):
                j = i + m - 1
                mid = i + (m - 1) // 2
                median = nums[mid]
                left_count = mid - i + 1
                right_count = j - mid
                left_sum = median * left_count - (prefix[mid + 1] - prefix[i])
                right_sum = (prefix[j + 1] - prefix[mid + 1]) - median * right_count
                total_cost = left_sum + right_sum
                if total_cost <= k:
                    return True
            return False
        
        low, high = 1, n
        max_freq = 1
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                max_freq = mid
                low = mid + 1
            else:
                high = mid - 1
        return max_freq