class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        max_freq = 1
        
        # Try each possible frequency from 2 to n
        for freq in range(2, n + 1):
            min_cost = float('inf')
            
            # Try each possible window of size 'freq'
            for start in range(n - freq + 1):
                end = start + freq - 1
                # Use median as target to minimize cost
                median_idx = start + freq // 2
                target = nums[median_idx]
                
                cost = 0
                for i in range(start, end + 1):
                    cost += abs(nums[i] - target)
                
                min_cost = min(min_cost, cost)
            
            if min_cost <= k:
                max_freq = freq
            else:
                break
        
        return max_freq