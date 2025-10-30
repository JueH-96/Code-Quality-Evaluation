class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # dpLeft[i][j] = maximum OR-value when choosing j elements
        # from nums[0..i] in order (subsequence)
        dpLeft = [[-1]*(k+1) for _ in range(n)]
        dpLeft[0][0] = 0
        dpLeft[0][1] = nums[0]
        
        for i in range(1, n):
            for j in range(k+1):
                # Option 1: do not take nums[i]
                dpLeft[i][j] = dpLeft[i-1][j]
                
                # Option 2: take nums[i], if possible
                if j > 0 and dpLeft[i-1][j-1] != -1:
                    candidate = dpLeft[i-1][j-1] | nums[i]
                    if candidate > dpLeft[i][j]:
                        dpLeft[i][j] = candidate
        
        # dpRight[i][j] = maximum OR-value when choosing j elements
        # from nums[i..n-1] in order (subsequence)
        dpRight = [[-1]*(k+1) for _ in range(n)]
        dpRight[n-1][0] = 0
        dpRight[n-1][1] = nums[n-1]
        
        for i in range(n-2, -1, -1):
            for j in range(k+1):
                # Option 1: do not take nums[i]
                dpRight[i][j] = dpRight[i+1][j]
                
                # Option 2: take nums[i], if possible
                if j > 0 and dpRight[i+1][j-1] != -1:
                    candidate = dpRight[i+1][j-1] | nums[i]
                    if candidate > dpRight[i][j]:
                        dpRight[i][j] = candidate
        
        # Now find the maximum (dpLeft[i][k] XOR dpRight[i+1][k])
        # where both sides are valid and 0 <= i < n-1
        # We must ensure i is large enough to choose k items on the left,
        # and that [i+1..end] is large enough to choose k on the right.
        # i ranges where dpLeft[i][k] != -1 and dpRight[i+1][k] != -1.
        
        answer = 0
        for i in range(k-1, n-k):
            left_val = dpLeft[i][k]
            right_val = dpRight[i+1][k]
            if left_val != -1 and right_val != -1:
                val = left_val ^ right_val
                if val > answer:
                    answer = val
        
        return answer