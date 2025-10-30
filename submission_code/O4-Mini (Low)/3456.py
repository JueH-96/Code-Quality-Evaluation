from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # dp is a dict mapping (last_value, used_changes) -> max subsequence length
        dp = {}
        ans = 0
        for x in nums:
            new_dp = dp.copy()
            # start a new subsequence with x
            if (x, 0) not in new_dp or new_dp[(x, 0)] < 1:
                new_dp[(x, 0)] = 1
                ans = max(ans, 1)
            # try to extend existing subsequences
            for (last, used), length in dp.items():
                # if same as last block value, no new change
                if x == last:
                    if new_dp.get((last, used), 0) < length + 1:
                        new_dp[(last, used)] = length + 1
                        ans = max(ans, length + 1)
                # if different, we need to "use" a change
                elif used < k:
                    if new_dp.get((x, used+1), 0) < length + 1:
                        new_dp[(x, used+1)] = length + 1
                        ans = max(ans, length + 1)
            dp = new_dp
        return ans