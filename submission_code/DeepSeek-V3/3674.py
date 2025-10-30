class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        invalid_subarrays = 0
        left = 0
        
        for right in range(1, n):
            # For each new element, compute the required operations to make the subarray from left to right non-decreasing
            # We need to find the left boundary such that the operations <= k
            # The operations required for a subarray nums[left..right] is sum_{i=left+1..right} max(0, nums[i-1] - nums[i])
            # But when moving right, the operations can be derived incrementally
            # However, when nums[right] < nums[right-1], we need to add (nums[right-1] - nums[right]) for each segment that affects it
            # So the approach is to use a sliding window where we track the required operations and adjust left accordingly
            pass
        
        # Implementing the sliding window approach
        res = 0
        left = 0
        total_ops = 0
        
        for right in range(n):
            if right > 0:
                # Calculate the required operations to fix nums[right] relative to nums[right-1]
                # The operations needed for the new element is the deficit if nums[right] < nums[right-1]
                # But this deficit might propagate back if previous elements were adjusted
                # So the total operations is the sum of (nums[i] - nums[i+1]) for i in left..right-1, but adjusted for previous increments
                # However, this is complex; instead, we can think of the operations as the sum of (max(0, required_prev - nums[i])) where required_prev is the required value of the previous element after operations
                pass
        
        # Alternative approach: for each subarray, the operations needed is the sum over i from left+1 to right of max(0, a[i-1] - a[i])
        # So sliding window where we maintain this sum <=k
        
        left = 0
        res = 0
        required_ops = 0
        
        for right in range(n):
            if right > 0:
                deficit = max(0, nums[right - 1] - nums[right])
                required_ops += deficit
                # Now, we need to move left forward until required_ops <=k
                while left < right and required_ops > k:
                    # When moving left forward, the deficits between left and left+1 are no longer considered
                    if left + 1 <= right:
                        removed_deficit = max(0, nums[left] - nums[left + 1])
                        required_ops -= removed_deficit
                    left += 1
            # The number of valid subarrays ending at right is right - left + 1
            res += right - left + 1
        
        return res