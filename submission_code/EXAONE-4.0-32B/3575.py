class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        L = [[set() for _ in range(k+1)] for _ in range(n)]
        L[0][0].add(0)
        if k >= 1:
            L[0][1].add(nums[0])
        
        for i in range(1, n):
            L[i][0].add(0)
            max_j = min(i+1, k)
            for j in range(1, max_j+1):
                s = set()
                s.update(L[i-1][j])
                for val in L[i-1][j-1]:
                    new_val = val | nums[i]
                    s.add(new_val)
                L[i][j] = s
        
        R = [[set() for _ in range(k+1)] for _ in range(n+1)]
        R[n][0].add(0)
        for j in range(1, k+1):
            R[n][j] = set()
        
        for i in range(n-1, -1, -1):
            R[i][0].add(0)
            max_j = min(n - i, k)
            for j in range(1, max_j+1):
                s = set()
                s.update(R[i+1][j])
                for val in R[i+1][j-1]:
                    new_val = val | nums[i]
                    s.add(new_val)
                R[i][j] = s
        
        ans = 0
        for m in range(k-1, n - k):
            left_set = L[m][k]
            right_set = R[m+1][k]
            for a in left_set:
                for b in right_set:
                    if a ^ b > ans:
                        ans = a ^ b
        return ans