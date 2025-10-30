from typing import List
import bisect

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Compute all pairwise differences
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(nums[j] - nums[i])
        if not diffs:
            # n<2 case won't happen as per constraints
            return 0
        sorted_diffs = sorted(diffs)
        
        # Build threshold list TL: D values at which f(D) may change.
        # These are D = 1, and D = d+1 for every actual diff d.
        TL = {1}
        for d in sorted_diffs:
            TL.add(d+1)
        # We only care about D up to max_diff+1
        max_diff = sorted_diffs[-1]
        TL = [d for d in TL if d <= max_diff+1]
        TL.sort()
        
        # DP function to compute f(D): number of k-subsets
        # with minimum adjacent difference >= D.
        def count_f(D: int) -> int:
            # prev[candidate last index i] = ways to pick c elements ending at i
            prev = [1] * n  # for c = 1
            # For c from 2 to k, build dp
            for c in range(2, k+1):
                # prefix sums of prev
                prefix = [0] * n
                s = 0
                for i in range(n):
                    s = (s + prev[i]) % mod
                    prefix[i] = s
                curr = [0] * n
                # compute curr[i] = sum prev[j] for j < i and nums[i]-nums[j] >= D
                for i in range(n):
                    # need j in [0..i-1] with nums[j] <= nums[i] - D
                    limit = nums[i] - D
                    # pos_end = bisect_right(nums, limit)
                    pos_end = bisect.bisect_right(nums, limit, 0, i)
                    j_max = pos_end - 1
                    if j_max >= 0:
                        curr[i] = prefix[j_max]
                prev = curr
                # early stop if no ways
                if all(x == 0 for x in prev):
                    return 0
            # total ways is sum of ways ending at any i
            return sum(prev) % mod
        
        ans = 0
        # Iterate thresholds TL[i], with next = TL[i+1], interval length = next - D
        for idx in range(len(TL) - 1):
            D = TL[idx]
            nextD = TL[idx+1]
            length = nextD - D
            if length <= 0:
                continue
            fD = count_f(D)
            ans = (ans + fD * length) % mod
        
        return ans