from typing import List
from array import array

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        if k == 0:                      # (the statement guarantees k ≥ 1, but keeps the code safe)
            return max(range(n))

        LOG = k.bit_length()            # enough levels to cover all bits of k

        # nexts[p][i]  : node reached from i after 2^p passes
        # sums[p][i]   : sum of ids along those 2^p passes (starting node NOT included)
        nexts = [array('I', receiver)]  # level 0 : one single pass
        sums  = [array('Q', receiver)]  # level 0 : sum of the one id we pass to

        # build doubling tables
        for _ in range(1, LOG):
            prev_next = nexts[-1]
            prev_sum  = sums[-1]

            nxt = array('I', [0] * n)
            sm  = array('Q', [0] * n)

            for i in range(n):
                mid       = prev_next[i]       # node after 2^(p-1) passes
                nxt[i]    = prev_next[mid]     # node after another 2^(p-1) passes
                sm[i]     = prev_sum[i] + prev_sum[mid]
            nexts.append(nxt)
            sums.append(sm)

        ans = 0
        # evaluate f(x) for every starting player x
        for i in range(n):
            cur = i
            total = i                          # include starting player's id
            steps = k
            p = 0
            while steps:
                if steps & 1:                  # take the segment of length 2^p
                    total += sums[p][cur]
                    cur = nexts[p][cur]
                steps >>= 1
                p += 1
            if total > ans:
                ans = total
        return ans