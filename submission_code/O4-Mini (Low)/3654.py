from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        total = sum(nums)
        # dp[j][l] = max benefit using j op1's and l op2's
        dp = [[-float('inf')] * (op2+1) for _ in range(op1+1)]
        dp[0][0] = 0
        
        for num in nums:
            # precompute benefits for this num
            # benefit of op1 alone
            half = (num + 1) // 2
            b1 = num - half
            # benefit of op2 alone (only if num >= k)
            b2 = k if num >= k else 0
            
            # benefit of both ops in the best order
            b_both = 0
            # try op1 then op2
            if half >= k:
                b1_then_2 = (num - half) + k
                b_both = max(b_both, b1_then_2)
            # try op2 then op1
            if num >= k:
                after2 = num - k
                half_after2 = (after2 + 1) // 2
                b2_then_1 = num - half_after2
                b_both = max(b_both, b2_then_1)
            
            # new dp for this iteration
            newdp = [[-float('inf')] * (op2+1) for _ in range(op1+1)]
            for j in range(op1+1):
                for l in range(op2+1):
                    cur = dp[j][l]
                    if cur < 0 and (j != 0 or l != 0):
                        continue
                    # no op
                    newdp[j][l] = max(newdp[j][l], cur)
                    # only op1
                    if j + 1 <= op1:
                        newdp[j+1][l] = max(newdp[j+1][l], cur + b1)
                    # only op2
                    if l + 1 <= op2 and num >= k:
                        newdp[j][l+1] = max(newdp[j][l+1], cur + b2)
                    # both ops
                    if j + 1 <= op1 and l + 1 <= op2 and b_both > 0:
                        newdp[j+1][l+1] = max(newdp[j+1][l+1], cur + b_both)
            dp = newdp
        
        # find best total benefit
        best = 0
        for j in range(op1+1):
            for l in range(op2+1):
                best = max(best, dp[j][l])
        
        return total - best