class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        
        # Prefix sum array for efficient range sum queries
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        def cost(left, right):
            # Calculate cost to make all elements in window equal to median
            mid = (left + right) // 2
            median = nums[mid]
            
            # Cost for elements left of median (need to increase them)
            left_sum = prefix[mid] - prefix[left]
            left_cost = median * (mid - left) - left_sum
            
            # Cost for elements right of median (need to decrease them)
            right_sum = prefix[right + 1] - prefix[mid + 1]
            right_cost = right_sum - median * (right - mid)
            
            return left_cost + right_cost
        
        max_freq = 1
        left = 0
        
        # Sliding window approach
        for right in range(n):
            # Shrink window while cost exceeds k
            while left < right and cost(left, right) > k:
                left += 1
            max_freq = max(max_freq, right - left + 1)
        
        return max_freq