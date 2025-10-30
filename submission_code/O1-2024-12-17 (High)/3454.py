class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        # d[i] will hold the difference target[i] - nums[i]
        d = [target[i] - nums[i] for i in range(n)]
        
        # We use the known formula:
        # Operations = ( |d[0]| + Σ|d[i] - d[i-1]| + |d[n-1]| ) / 2
        # (where d[-1] is considered 0).
        # This counts how many total "steps" of +1/-1 across subarray boundaries
        # are needed to bring all differences to zero.
        
        total = abs(d[0])
        for i in range(1, n):
            total += abs(d[i] - d[i - 1])
        total += abs(d[n - 1])
        
        return total // 2