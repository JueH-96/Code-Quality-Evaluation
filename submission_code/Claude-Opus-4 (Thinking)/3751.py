class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)
        initial_k_count = nums.count(k)
        max_freq = initial_k_count
        
        # Try converting each possible value v to k
        for v in range(1, 51):  # Given constraint: 1 <= nums[i] <= 50
            if v == k:
                continue  # No need to convert k to k
            
            # Use Kadane's algorithm to find the best subarray
            max_gain = 0
            current_gain = 0
            
            for num in nums:
                if num == v:
                    current_gain += 1
                elif num == k:
                    current_gain -= 1
                
                if current_gain < 0:
                    current_gain = 0
                
                max_gain = max(max_gain, current_gain)
            
            # The maximum frequency of k when converting v to k
            max_freq = max(max_freq, initial_k_count + max_gain)
        
        return max_freq