class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_diff = float('inf')
        
        for i in range(n):
            or_result = 0
            for j in range(i, n):
                or_result |= nums[j]
                diff = abs(k - or_result)
                min_diff = min(min_diff, diff)
                
                # If we find a perfect match, return immediately
                if diff == 0:
                    return 0
        
        return min_diff