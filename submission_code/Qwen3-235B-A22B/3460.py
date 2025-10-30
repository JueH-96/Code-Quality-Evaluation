from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        required = {}
        for end, cnt in requirements:
            m = end + 1
            required[m] = cnt
        
        for m in required:
            if m < 0 or m > n:
                return 0
        if n not in required:
            return 0
        
        current_dp = [1]  # m=0, which has one permutation (empty) with 0 inversions
        
        for m in range(n):
            # Transitioning from size m to m+1
            current_size = m
            next_len = m + 1
            
            if next_len in required:
                k_needed = required[next_len]
                if k_needed < 0 or k_needed >= len(current_dp):
                    return 0
                val = current_dp[k_needed] % MOD
                new_dp = [0] * (k_needed + 1)
                new_dp[k_needed] = val
                current_dp = new_dp
            
            # Compute transition to size m+1
            len_dp = len(current_dp)
            prefix = [0] * (len_dp + 1)
            for i in range(len_dp):
                prefix[i + 1] = (prefix[i] + current_dp[i]) % MOD
            
            new_max_j = current_size * (current_size + 1) // 2
            new_dp = [0] * (new_max_j + 1)
            
            for j in range(len(new_dp)):
                low = max(0, j - current_size)
                high = min(j, len_dp - 1)
                if low > high:
                    new_dp[j] = 0
                else:
                    sum_val = (prefix[high + 1] - prefix[low]) % MOD
                    new_dp[j] = sum_val if sum_val >= 0 else sum_val + MOD
            
            current_dp = new_dp
        
        return current_dp[required[n]] % MOD