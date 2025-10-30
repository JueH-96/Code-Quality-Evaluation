from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the rewards in non‑decreasing order
        arr = sorted(rewardValues)
        maxV = arr[-1]
        # We'll keep a bitset `cur` of length maxV+1 where
        # cur & (1<<s) != 0 means "there exists a pickable subset of
        # processed items whose sum is exactly s".
        # We only need to track sums up to maxV, since any sum >=
        # next reward value can't help us pick it.
        mask = (1 << (maxV + 1)) - 1
        cur = 1  # only sum=0 is possible initially
        ans = 0

        for v in arr:
            # Find the largest s < v for which cur has bit s set.
            # That s is the best total we can have before picking v.
            possible = cur & ((1 << v) - 1)
            s = possible.bit_length() - 1  # highest set bit < v
            ans = max(ans, s + v)

            # Update the bitset to include sums that use v:
            cur = (cur | (cur << v)) & mask

        return ans