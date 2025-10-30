from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Compute OR of all numbers
        OR_all = 0
        for x in nums:
            OR_all |= x
        # Prepare inverted numbers (only lower 30 bits matter)
        mask_full = (1 << 30) - 1
        inv_nums = [mask_full ^ x for x in nums]  # this is (~x) & mask_full

        disallowed = 0  # bits we have managed to forbid in the final OR

        # Helper to check if we can forbid all bits in 'forbidden_mask'
        # using at most k merges (i.e. dp2[0] <= k).
        def feasible(forbidden_mask: int) -> bool:
            # If no forbidden bits, trivially okay
            if forbidden_mask == 0:
                return True

            # Count how many bits we must cover with zeros in each segment
            popcnt = forbidden_mask.bit_count()
            # We'll build minimal_j[i]: the minimal end index j >= i
            # so that the window [i..j] contains at least one zero
            # for each bit in forbidden_mask.
            n_local = n
            counts = [0] * 30      # counts[b] = how many zeros for bit b in window
            distinct = 0           # how many forbidden bits currently covered
            minimal_j = [n_local] * n_local
            j = 0

            # Two‐pointer sweep to build minimal_j
            for i in range(n_local):
                # expand j until we cover all forbidden bits
                while j < n_local and distinct < popcnt:
                    zeros = inv_nums[j] & forbidden_mask
                    temp = zeros
                    # for each bit in zeros, update counts
                    while temp:
                        lb = temp & -temp
                        bpos = lb.bit_length() - 1
                        if counts[bpos] == 0:
                            distinct += 1
                        counts[bpos] += 1
                        temp ^= lb
                    j += 1
                # record minimal end
                if distinct == popcnt:
                    minimal_j[i] = j - 1
                else:
                    minimal_j[i] = n_local
                # before moving i forward, remove i's contribution
                zeros_i = inv_nums[i] & forbidden_mask
                temp = zeros_i
                while temp:
                    lb = temp & -temp
                    bpos = lb.bit_length() - 1
                    counts[bpos] -= 1
                    if counts[bpos] == 0:
                        distinct -= 1
                    temp ^= lb

            # DP to compute minimal merges needed for suffix starting at i:
            # dp2[i] = minimal merges to partition nums[i..n-1] into
            # valid segments. We check dp2[0] <= k.
            INF = n_local + 1
            dp2 = [INF] * (n_local + 1)
            dp2[n_local] = 0
            # suff_min[j] = min_{t >= j} ( t + dp2[t+1] )
            suff_min = [INF] * (n_local + 1)
            suff_min[n_local] = INF

            # build dp2 and suff_min backwards
            for idx in range(n_local - 1, -1, -1):
                # update suffix‐minimum
                val = idx + dp2[idx + 1]
                if val < suff_min[idx + 1]:
                    suff_min[idx] = val
                else:
                    suff_min[idx] = suff_min[idx + 1]
                # if we have a valid minimal_j, use it
                mj = minimal_j[idx]
                if mj < n_local:
                    # dp2[idx] = min_{j >= mj} [ (j - idx) + dp2[j+1] ]
                    #           = -idx + min_{j>=mj}[ j + dp2[j+1] ]
                    dp2[idx] = suff_min[mj] - idx
                # else dp2[idx] stays INF

            # check if total merges needed <= k
            return dp2[0] <= k

        # Greedily try to forbid high bits first
        for b in range(29, -1, -1):
            bit = 1 << b
            # skip if this bit never appears
            if (OR_all & bit) == 0:
                continue
            new_mask = disallowed | bit
            if feasible(new_mask):
                disallowed = new_mask

        # The answer is OR_all with all disallowed bits cleared
        return OR_all & (~disallowed)