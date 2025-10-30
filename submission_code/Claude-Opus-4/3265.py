class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')
        found = False
        
        # Create a dictionary to store all indices for each value
        value_indices = {}
        for i in range(n):
            if nums[i] not in value_indices:
                value_indices[nums[i]] = []
            value_indices[nums[i]].append(i)
        
        # Calculate prefix sums for efficient subarray sum calculation
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # For each starting position
        for i in range(n):
            # Check for nums[i] + k
            target1 = nums[i] + k
            if target1 in value_indices:
                for j in value_indices[target1]:
                    if j > i:  # j must be after i
                        found = True
                        # Sum from i to j inclusive
                        current_sum = prefix_sum[j + 1] - prefix_sum[i]
                        max_sum = max(max_sum, current_sum)
            
            # Check for nums[i] - k
            target2 = nums[i] - k
            if target2 in value_indices:
                for j in value_indices[target2]:
                    if j > i:  # j must be after i
                        found = True
                        # Sum from i to j inclusive
                        current_sum = prefix_sum[j + 1] - prefix_sum[i]
                        max_sum = max(max_sum, current_sum)
        
        return max_sum if found else 0