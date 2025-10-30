class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Precompute the scores for all possible subarrays
        scores = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                current = nums[i:j+1]
                while len(current) > 1:
                    new_current = []
                    for k in range(len(current) - 1):
                        new_current.append(current[k] ^ current[k+1])
                    current = new_current
                if current:
                    scores[i][j] = current[0]
                else:
                    scores[i][j] = 0
        
        # Process each query
        result = []
        for L, R in queries:
            max_score = 0
            for i in range(L, R + 1):
                for j in range(i, R + 1):
                    if scores[i][j] > max_score:
                        max_score = scores[i][j]
            result.append(max_score)
        return result