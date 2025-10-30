class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_or = [0] * (n + 1)
        for i in range(n):
            prefix_or[i + 1] = prefix_or[i] | nums[i]
        
        min_diff = float('inf')
        for i in range(n):
            for j in range(i, n):
                subarray_or = prefix_or[j + 1] ^ prefix_or[i]
                min_diff = min(min_diff, abs(k - subarray_or))
        
        return min_diff