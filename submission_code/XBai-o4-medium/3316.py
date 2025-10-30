from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        
        # Generate all unique pairwise differences
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(abs(nums[i] - nums[j]))
        
        # Sort the differences and add a sentinel value
        sorted_diffs = sorted(diffs)
        sorted_diffs.append(float('inf'))  # Sentinel value
        
        total = 0
        
        for i in range(len(sorted_diffs)):
            current_D = sorted_diffs[i]
            # Determine next_D
            if i + 1 < len(sorted_diffs):
                next_D = sorted_diffs[i + 1]
            else:
                next_D = float('inf')
            
            # Compute f(current_D)
            f_current = self.count_subseq(nums, current_D, k)
            
            # Compute f_next_D
            if i + 1 < len(sorted_diffs):
                f_next = self.count_subseq(nums, next_D, k)
            else:
                f_next = 0
            
            contribution = current_D * (f_current - f_next)
            total += contribution
        
        return total % MOD
    
    def count_subseq(self, A: List[int], D: float, k: int) -> int:
        n = len(A)
        # dp[i][j] represents the number of ways to select j elements from first i elements
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1  # Base case: 1 way to select 0 elements from 0 elements
        
        for i in range(1, n + 1):
            current_val = A[i - 1]
            for j in range(k + 1):
                # Copy the count from the previous state (not taking the current element)
                dp[i][j] = dp[i - 1][j]
                
                if j >= 1:
                    target = current_val - D
                    low, high = 0, i - 2
                    pos = -1
                    # Binary search to find the largest index with value <= target
                    while low <= high:
                        mid = (low + high) // 2
                        if A[mid] <= target:
                            pos = mid
                            low = mid + 1
                        else:
                            high = mid - 1
                    # Add the number of ways from the found position
                    if pos + 1 >= 0:
                        dp[i][j] += dp[pos + 1][j - 1]
        
        return dp[n][k]