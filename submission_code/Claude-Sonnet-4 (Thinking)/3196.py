class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        def cost_to_make_equal(start, end):
            # Cost to make nums[start:end+1] all equal to the median
            length = end - start + 1
            mid = start + length // 2
            median = nums[mid]
            
            # Sum of |nums[i] - median| for i in [start, end]
            left_count = mid - start
            left_sum = prefix[mid] - prefix[start]
            left_cost = left_count * median - left_sum
            
            right_count = end - mid
            right_sum = prefix[end + 1] - prefix[mid + 1]
            right_cost = right_sum - right_count * median
            
            return left_cost + right_cost
        
        def can_achieve_frequency(target_freq):
            min_cost = float('inf')
            for i in range(n - target_freq + 1):
                cost = cost_to_make_equal(i, i + target_freq - 1)
                min_cost = min(min_cost, cost)
            return min_cost <= k
        
        # Binary search on the answer
        left, right = 1, n
        result = 1
        
        while left <= right:
            mid = (left + right) // 2
            if can_achieve_frequency(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result