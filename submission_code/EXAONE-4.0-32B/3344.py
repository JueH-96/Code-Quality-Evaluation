class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        A = [p[0] + p[1] for p in points]
        B = [p[0] - p[1] for p in points]
        
        prefix_max_A = [0] * n
        prefix_min_A = [0] * n
        prefix_max_A[0] = A[0]
        prefix_min_A[0] = A[0]
        for i in range(1, n):
            prefix_max_A[i] = max(prefix_max_A[i-1], A[i])
            prefix_min_A[i] = min(prefix_min_A[i-1], A[i])
        
        suffix_max_A = [0] * n
        suffix_min_A = [0] * n
        suffix_max_A[-1] = A[-1]
        suffix_min_A[-1] = A[-1]
        for i in range(n-2, -1, -1):
            suffix_max_A[i] = max(suffix_max_A[i+1], A[i])
            suffix_min_A[i] = min(suffix_min_A[i+1], A[i])
        
        prefix_max_B = [0] * n
        prefix_min_B = [0] * n
        prefix_max_B[0] = B[0]
        prefix_min_B[0] = B[0]
        for i in range(1, n):
            prefix_max_B[i] = max(prefix_max_B[i-1], B[i])
            prefix_min_B[i] = min(prefix_min_B[i-1], B[i])
        
        suffix_max_B = [0] * n
        suffix_min_B = [0] * n
        suffix_max_B[-1] = B[-1]
        suffix_min_B[-1] = B[-1]
        for i in range(n-2, -1, -1):
            suffix_max_B[i] = max(suffix_max_B[i+1], B[i])
            suffix_min_B[i] = min(suffix_min_B[i+1], B[i])
        
        ans = float('inf')
        for j in range(n):
            if j == 0:
                maxA = suffix_max_A[1]
                minA = suffix_min_A[1]
                maxB = suffix_max_B[1]
                minB = suffix_min_B[1]
            elif j == n-1:
                maxA = prefix_max_A[n-2]
                minA = prefix_min_A[n-2]
                maxB = prefix_max_B[n-2]
                minB = prefix_min_B[n-2]
            else:
                maxA = max(prefix_max_A[j-1], suffix_max_A[j+1])
                minA = min(prefix_min_A[j-1], suffix_min_A[j+1])
                maxB = max(prefix_max_B[j-1], suffix_max_B[j+1])
                minB = min(prefix_min_B[j-1], suffix_min_B[j+1])
            
            rangeA = maxA - minA
            rangeB = maxB - minB
            candidate = max(rangeA, rangeB)
            if candidate < ans:
                ans = candidate
                
        return ans