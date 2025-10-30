from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Maximum value in nums
        max_val = max(nums)
        
        # Frequency of each number in nums
        freq = [0] * (max_val + 1)
        for x in nums:
            freq[x] += 1
        
        # cnt_div[d] = count of numbers in nums divisible by d
        cnt_div = [0] * (max_val + 1)
        for d in range(1, max_val + 1):
            for m in range(d, max_val + 1, d):
                cnt_div[d] += freq[m]
        
        # pairs_div[d] = number of pairs (i,j) with both nums[i] and nums[j] divisible by d
        pairs_div = [0] * (max_val + 1)
        for d in range(1, max_val + 1):
            c = cnt_div[d]
            pairs_div[d] = c * (c - 1) // 2
        
        # pairs_gcd[d] = number of pairs whose gcd is exactly d
        # Start with pairs_div and subtract out multiples
        pairs_gcd = pairs_div[:]  # copy
        for d in range(max_val, 0, -1):
            multiple = 2 * d
            while multiple <= max_val:
                pairs_gcd[d] -= pairs_gcd[multiple]
                multiple += d
        
        # We now have for each d, pairs_gcd[d] = count of pairs with gcd == d.
        # We need to answer queries: for each query q, find the q-th element
        # in the sorted list of all gcd-pairs (0-based).
        
        # Sort queries by their index, keep original positions
        qlist = sorted((q, i) for i, q in enumerate(queries))
        answers = [0] * len(queries)
        
        # Sweep d from smallest gcd to largest, maintaining prefix sums of counts
        cum = 0
        qi = 0
        for d in range(1, max_val + 1):
            count_here = pairs_gcd[d]
            if count_here <= 0:
                continue
            # while the next query index falls in [cum, cum + count_here)
            while qi < len(qlist) and qlist[qi][0] < cum + count_here:
                q_idx, orig_i = qlist[qi]
                answers[orig_i] = d
                qi += 1
            cum += count_here
            if qi >= len(qlist):
                break
        
        return answers