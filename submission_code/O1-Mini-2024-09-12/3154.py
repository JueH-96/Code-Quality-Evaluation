class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        # Precompute max_after[j] = max(nums[k] for k > j)
        max_after = [0] * n
        max_after[-1] = nums[-1]
        for j in range(n-2, -1, -1):
            max_after[j] = max(nums[j], max_after[j+1])
        
        max_triplet = float('-inf')
        max_before = nums[0]
        
        for j in range(1, n-1):
            # Update max_before
            max_before = max(max_before, nums[j-1])
            current_value = (max_before - nums[j]) * max_after[j+1]
            max_triplet = max(max_triplet, current_value)
        
        return max_triplet if max_triplet > 0 else 0