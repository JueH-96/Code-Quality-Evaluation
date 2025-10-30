from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2
        max_x = k
        
        count_0 = [0] * (k + 2)  # X ranges from 0 to k
        freq_S = [0] * (k + 2)   # S ranges from 0 to k
        
        for i in range(m):
            a = nums[i]
            b = nums[n - 1 - i]
            D = abs(a - b)
            count_0[D] += 1
            
            va = max(a, k - a)
            vb = max(b, k - b)
            s_i = max(va, vb)
            if s_i <= k:
                freq_S[s_i] += 1
            else:
                # This case shouldn't happen because a and b are between 0 and k
                pass  # Handle if necessary, but per problem constraints, s_i <=k
        
        # Compute suffix_sum array
        suffix_sum = [0] * (k + 2)  # indexes 0 to k+1
        suffix_sum[k + 1] = 0
        for x in range(k, -1, -1):
            suffix_sum[x] = suffix_sum[x + 1] + freq_S[x]
        
        max_savings = 0
        for X in range(k + 1):
            current_savings = count_0[X] + suffix_sum[X]
            if current_savings > max_savings:
                max_savings = current_savings
        
        return 2 * m - max_savings