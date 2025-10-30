class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        # Precompute prefix maximum: for each j, best i in [0, j-1]
        prefix_max = [0] * n
        prefix_max[0] = nums[0]
        for j in range(1, n):
            prefix_max[j] = max(prefix_max[j-1], nums[j])
            
        # Precompute suffix maximum: for each j, best k in [j+1, n-1]
        suffix_max = [0] * n
        suffix_max[n-1] = nums[n-1]
        for j in range(n-2, -1, -1):
            suffix_max[j] = max(suffix_max[j+1], nums[j])
        
        best = 0
        # j goes from 1 to n-2, so that i < j < k always exists
        for j in range(1, n-1):
            diff = prefix_max[j-1] - nums[j]
            # Only consider if difference is non-negative (otherwise product negative)
            if diff <= 0:
                continue
            product = diff * suffix_max[j+1]
            best = max(best, product)
        
        return best