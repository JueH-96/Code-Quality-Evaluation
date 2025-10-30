class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def max_non_adjacent_sum(arr):
            if not arr:
                return 0
            n = len(arr)
            if n == 1:
                return max(0, arr[0])
            prev1, prev2 = max(0, arr[0]), 0
            for i in range(1, n):
                current = max(prev1, prev2 + arr[i])
                prev2 = prev1
                prev1 = current
            return prev1
        
        total_sum = 0
        
        for pos, x in queries:
            nums[pos] = x
            total_sum = (total_sum + max_non_adjacent_sum(nums)) % MOD
        
        return total_sum