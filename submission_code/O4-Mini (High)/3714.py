from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Cj[j] = C(i, j) for the current i
        Cj = [1] + [0] * (k - 1)
        sum_low = [0] * n
        sum_Cj = 1  # sum of Cj[0..k-1] for the current i
        
        # Build sum_low[i] = sum_{j=0..k-1} C(i, j) for i=0..n-1
        for i in range(n):
            sum_low[i] = sum_Cj
            old_last = Cj[k - 1]  # C(i, k-1)
            # update Cj in place to represent C(i+1, j)
            for j in range(k - 1, 0, -1):
                Cj[j] += Cj[j - 1]
                if Cj[j] >= MOD:
                    Cj[j] -= MOD
            # Cj[0] stays 1
            # update the sum accordingly: new_sum = sum_Cj + sum_{j=0..k-2} C(i, j)
            # which equals sum_Cj + (sum_Cj - old_last)
            sum_Cj = (sum_Cj + (sum_Cj - old_last)) % MOD
        
        # Now accumulate the answer using sum_low for min and max contributions
        ans = 0
        for i, v in enumerate(nums):
            ans = (ans
                   + v * sum_low[i]           # v as max in subsets
                   + v * sum_low[n - 1 - i]   # v as min in subsets
                   ) % MOD
        
        return ans