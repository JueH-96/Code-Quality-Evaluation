class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        b_sorted = sorted(enumerate(b), key=lambda x: x[1])
        
        max_score = 0
        for i in range(n-3):
            for j in range(i+1, n-2):
                for k in range(j+1, n-1):
                    for l in range(k+1, n):
                        score = a[0] * b_sorted[i][1] + a[1] * b_sorted[j][1] + a[2] * b_sorted[k][1] + a[3] * b_sorted[l][1]
                        max_score = max(max_score, score)
        
        return max_score