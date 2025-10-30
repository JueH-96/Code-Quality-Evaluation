class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_freq = 0
        
        # Try all possible subarrays [i, j]
        for i in range(n):
            for j in range(i, n):
                # For this subarray, we need to choose x to maximize frequency of k
                # Count frequency of each (k - nums[m]) for m in [i, j]
                x_count = {}
                for m in range(i, j + 1):
                    x_needed = k - nums[m]
                    x_count[x_needed] = x_count.get(x_needed, 0) + 1
                
                # Try each possible x value
                for x, count_in_subarray in x_count.items():
                    total_freq = 0
                    
                    # Count k's outside the subarray (unchanged)
                    for idx in range(n):
                        if idx < i or idx > j:
                            if nums[idx] == k:
                                total_freq += 1
                    
                    # Add count of elements in subarray that become k
                    total_freq += count_in_subarray
                    
                    max_freq = max(max_freq, total_freq)
        
        return max_freq