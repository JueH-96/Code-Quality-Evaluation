class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n <= 1:
            return n
        
        nums.sort()
        max_freq = 1
        
        # Iterate through all unique elements as potential targets
        i = 0
        while i < n:
            target = nums[i]
            
            # Count initial frequency of the target
            j = i
            while j < n and nums[j] == target:
                j += 1
            
            # Initialize pointers for expansion
            left = i - 1
            right = j
            ops_used = 0
            freq = j - i  # Initial frequency of the target
            
            # Expand left and right as much as possible using at most k operations
            while True:
                left_cost = float('inf') if left < 0 else target - nums[left]
                right_cost = float('inf') if right >= n else nums[right] - target
                
                # If we can't expand in either direction, break
                if min(left_cost, right_cost) == float('inf') or (ops_used + min(left_cost, right_cost) > k):
                    break
                
                # Choose the direction with lower cost
                if left_cost <= right_cost:
                    ops_used += left_cost
                    freq += 1
                    left -= 1
                else:
                    ops_used += right_cost
                    freq += 1
                    right += 1
            
            max_freq = max(max_freq, freq)
            i = j  # Skip all duplicates
        
        return max_freq