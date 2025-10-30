class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        max_score = float('-inf')
        
        # Try all possible combinations of 4 indices
        for i in range(n-3):
            for j in range(i+1, n-2):
                for k in range(j+1, n-1):
                    for l in range(k+1, n):
                        score = a[0] * b[i] + a[1] * b[j] + a[2] * b[k] + a[3] * b[l]
                        max_score = max(max_score, score)
                        
        return max_score