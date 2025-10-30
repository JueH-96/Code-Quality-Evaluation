from typing import List

MOD = 10 ** 9 + 7

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)

        # dp for the first position : every value 0..nums[0] is possible
        prev = [1] * (nums[0] + 1)          # prev[v] = ways with arr1[last] == v

        for i in range(1, n):
            # minimal increase that arr1[i] must have compared with arr1[i-1]
            inc = max(0, nums[i] - nums[i - 1])

            # prefix sums of prev, prefix[x] = sum_{u=0..x} prev[u]  (mod MOD)
            prefix = [0] * len(prev)
            running = 0
            for v, cnt in enumerate(prev):
                running = (running + cnt) % MOD
                prefix[v] = running

            # compute current dp
            cur = [0] * (nums[i] + 1)       # values larger than nums[i] are impossible
            for v in range(inc, nums[i] + 1):
                # we can pick any previous value u such that
                #    u ≤ v - inc        (see analysis)
                idx = v - inc
                if idx >= len(prefix):      # this cannot really happen, safety check
                    idx = len(prefix) - 1
                cur[v] = prefix[idx]        # number of sequences ending with arr1[i]=v

            prev = cur                      # move to next position

        return sum(prev) % MOD