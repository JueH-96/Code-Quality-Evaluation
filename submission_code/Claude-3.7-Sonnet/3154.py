class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_val = 0  # Final answer
        
        # max_before[j] represents the maximum value of nums[i] for all i < j
        max_before = [0] * n
        for j in range(1, n):
            max_before[j] = max(max_before[j-1], nums[j-1])
        
        # Keep track of the maximum difference (nums[i] - nums[j]) seen so far
        max_diff = 0
        for k in range(2, n):
            # Update the maximum difference before position k
            max_diff = max(max_diff, max_before[k-1] - nums[k-1])
            
            # Calculate the value of the triplet and update max_val
            val = max_diff * nums[k]
            max_val = max(max_val, val)
        
        return max_val