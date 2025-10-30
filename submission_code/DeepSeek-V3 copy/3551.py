class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # Precompute the XOR scores for all possible subarrays
        n = len(nums)
        xor_scores = [[0] * n for _ in range(n)]
        
        for i in range(n):
            xor_scores[i][i] = nums[i]
            for j in range(i+1, n):
                xor_scores[i][j] = xor_scores[i][j-1] ^ nums[j]
        
        # For each query, find the maximum XOR score in the specified range
        answer = []
        for l, r in queries:
            max_xor = 0
            for i in range(l, r+1):
                for j in range(i, r+1):
                    max_xor = max(max_xor, xor_scores[i][j])
            answer.append(max_xor)
        
        return answer