class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Compute XOR scores for all subarrays
        xor_score = [[0] * n for _ in range(n)]
        for i in range(n):
            xor_score[i][i] = nums[i]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                xor_score[i][j] = xor_score[i][j-1] ^ xor_score[i+1][j]
        
        # Compute maximum XOR scores for all ranges
        max_score = [[0] * n for _ in range(n)]
        for i in range(n):
            max_score[i][i] = nums[i]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                max_score[i][j] = max(max_score[i][j-1], max_score[i+1][j], xor_score[i][j])
        
        # Answer queries
        result = []
        for l, r in queries:
            result.append(max_score[l][r])
        
        return result