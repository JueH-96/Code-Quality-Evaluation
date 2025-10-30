class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_diff = float('inf')
        
        # For each starting index
        for i in range(n):
            curr_or = 0
            # Try all possible subarrays starting at i
            for j in range(i, n):
                curr_or |= nums[j]
                min_diff = min(min_diff, abs(k - curr_or))
                
                # If we found a perfect match, return 0
                if min_diff == 0:
                    return 0
                
                # If current OR is already greater than k, no need to extend further
                if curr_or > k and curr_or - k >= min_diff:
                    break
                    
        return min_diff