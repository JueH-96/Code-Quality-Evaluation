from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # coefficient for the j-th chosen sub-array  (1-based)
        # c(j) =  (+) (k-j+1)  if j is odd
        #       = (−) (k-j+1)  if j is even
        coeff = [0] * (k + 1)
        for j in range(1, k + 1):
            coeff[j] = (k - j + 1) * (1 if j & 1 else -1)

        NEG = -10 ** 30                     # sufficiently small sentinel
        dp_not = [NEG] * (k + 1)            # finished j sub-arrays, not inside one
        dp_in  = [NEG] * (k + 1)            # currently inside the j-th sub-array
        dp_not[0] = 0                       # nothing chosen yet

        for val in nums:
            # copy of dp_not (doing nothing at this index is always allowed)
            dp_not_next = dp_not[:]         
            dp_in_next  = [NEG] * (k + 1)

            # 1) continue current sub-arrays or close them at this position
            for j in range(1, k + 1):
                if dp_in[j] == NEG:
                    continue
                cont = dp_in[j] + val * coeff[j]          # extend j-th sub-array
                if cont > dp_in_next[j]:
                    dp_in_next[j] = cont                  # keep it open
                if cont > dp_not_next[j]:
                    dp_not_next[j] = cont                 # close it here

            # 2) start a new sub-array at this position (length ≥ 1)
            for j in range(0, k):                         # j already finished
                if dp_not[j] == NEG:
                    continue
                start = dp_not[j] + val * coeff[j + 1]    # open (j+1)-th
                if start > dp_in_next[j + 1]:
                    dp_in_next[j + 1] = start             # leave it open
                if start > dp_not_next[j + 1]:
                    dp_not_next[j + 1] = start            # close immediately

            dp_not, dp_in = dp_not_next, dp_in_next

        # we must have all k sub-arrays closed
        return dp_not[k]