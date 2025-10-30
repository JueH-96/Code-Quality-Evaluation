class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute the maximum OR for each starting index i
        max_or = [0] * n
        current_max = 0
        for i in range(n-1, -1, -1):
            current_max |= nums[i]
            max_or[i] = current_max
        
        min_diff = float('inf')
        for i in range(n):
            current_or = 0
            for j in range(i, n):
                current_or |= nums[j]
                if current_or >= max_or[i]:
                    break
                diff = abs(k - current_or)
                if diff < min_diff:
                    min_diff = diff
        return min_diff