from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        MAX_BIT = 7          # because nums[i] < 2**7
        VALS = 1 << MAX_BIT  # 128
        
        # helper : given a bit-mask M (128 bits) that encodes a set of numbers,
        #          return another mask that contains (v | x) for every v in the set
        def apply_or(mask: int, x: int) -> int:
            res = 0
            while mask:
                low = mask & -mask
                v   = low.bit_length() - 1
                res |= 1 << (v | x)
                mask ^= low
            return res
        
        # ------------------  SUFFIX part : all possibilities for the right half  ---------
        # suffix_k[i]  – bit-mask of OR values obtainable by choosing exactly k elements
        #                inside the sub-array nums[i .. n-1]  (order preserved)
        suffix_k = [0] * (n + 1)
        dp = [0] * (k + 1)          # dp[s] – mask for picking s elements in current suffix
        dp[0] = 1                   # choose nothing => OR value 0
        
        for i in range(n - 1, -1, -1):
            new_dp = dp[:]          # we may skip nums[i]
            val = nums[i]
            for s in range(k - 1, -1, -1):  # add nums[i] to selections of size s
                if dp[s]:
                    new_dp[s + 1] |= apply_or(dp[s], val)
            dp = new_dp
            suffix_k[i] = dp[k]     # record possibilities for this starting index
        
        # ------------------  PREFIX scan : build the left half on the fly  ---------------
        pre_dp = [0] * (k + 1)      # same meaning as dp but for prefix
        pre_dp[0] = 1
        
        ans = 0
        for i in range(n + 1):      # i is the boundary – first k elements are before i
            left_mask  = pre_dp[k]          # OR results achievable for the first half
            right_mask = suffix_k[i]        # OR results achievable for the second half
            if left_mask and right_mask:    # both halves possible
                # enumerate every pair (o1, o2) and maximise XOR
                tmp_left = left_mask
                while tmp_left:
                    low1 = tmp_left & -tmp_left
                    o1   = low1.bit_length() - 1
                    tmp_left ^= low1
                    
                    tmp_right = right_mask
                    while tmp_right:
                        low2 = tmp_right & -tmp_right
                        o2   = low2.bit_length() - 1
                        tmp_right ^= low2
                        xor_val = o1 ^ o2
                        if xor_val > ans:
                            ans = xor_val
                            
            if i == n:          # no more elements – finished
                break
            
            # extend prefix with nums[i] for subsequent boundaries
            val = nums[i]
            new_pre = pre_dp[:]
            for s in range(k - 1, -1, -1):
                if pre_dp[s]:
                    new_pre[s + 1] |= apply_or(pre_dp[s], val)
            pre_dp = new_pre
        
        return ans