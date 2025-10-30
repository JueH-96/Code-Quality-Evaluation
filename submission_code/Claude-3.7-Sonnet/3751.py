class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        # Initial count of k in the array
        init_count = nums.count(k)
        max_frequency = init_count
        
        # Try all possible values to add
        # Based on constraints: 1 ≤ nums[i] ≤ 50, 1 ≤ k ≤ 50
        for x in range(k - 50, k + 1):
            # Create a modified array to track the effect of adding x
            mod = [0] * len(nums)
            for i in range(len(nums)):
                if nums[i] != k and nums[i] + x == k:
                    # Adding x to this position creates a new k
                    mod[i] = 1
                elif nums[i] == k and nums[i] + x != k:
                    # Adding x to this position removes an existing k
                    mod[i] = -1
            
            # Use Kadane's algorithm to find maximum subarray sum
            # This represents the optimal net change in k's count
            max_sum = 0
            current_sum = 0
            
            for val in mod:
                current_sum = max(0, current_sum + val)
                max_sum = max(max_sum, current_sum)
            
            # Calculate total frequency after operation
            frequency = init_count + max_sum
            max_frequency = max(max_frequency, frequency)
        
        return max_frequency