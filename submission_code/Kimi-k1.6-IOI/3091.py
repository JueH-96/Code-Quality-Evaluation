from collections import Counter
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        freq = Counter(nums)
        count_zero = freq.get(0, 0)
        
        # Remove zeros from the frequency count
        if 0 in freq:
            del freq[0]
        
        total_sum = sum(nums)
        max_needed = min(r, total_sum)
        if max_needed < 0:
            return 0
        
        # Initialize dp array
        dp = [0] * (max_needed + 1)
        dp[0] = 1
        
        for x in freq:
            cnt = freq[x]
            tmp = dp.copy()
            for residue in range(x):
                s_list = []
                s = residue
                while s <= max_needed:
                    s_list.append(s)
                    s += x
                if not s_list:
                    continue
                # Generate arr and prefix sums
                arr = [tmp[s] for s in s_list]
                prefix = [0] * (len(arr) + 1)
                for i in range(len(arr)):
                    prefix[i + 1] = (prefix[i] + arr[i]) % MOD
                # Update dp for each s in s_list
                for idx in range(len(s_list)):
                    s_val = s_list[idx]
                    m = idx
                    max_k = min(cnt, m)
                    if max_k == 0:
                        continue
                    start = idx - max_k
                    sum_terms = (prefix[idx] - prefix[max(0, start)]) % MOD
                    dp[s_val] = (dp[s_val] + sum_terms) % MOD
        
        min_sum = max(l, 0)
        max_sum = max_needed
        if max_sum < min_sum:
            return 0
        res = sum(dp[min_sum : max_sum + 1]) % MOD
        res = res * (count_zero + 1) % MOD
        return res