from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        # Quick check: if |k| exceeds the maximum possible alternating sum, impossible
        maxSum = sum(nums)
        if k > maxSum or k < -maxSum:
            return -1

        # dpAll[(s, p)] = max product (<= limit) to reach alternating-sum = s
        #                  with parity p (0 = next pick is even-index in subseq, 1 = odd)
        # dpNonEmpty tracks the same but only for non-empty subsequences
        dpAll = {(0, 0): 1}
        dpNonEmpty = {}

        for num in nums:
            newAll = dpAll.copy()
            newNe = dpNonEmpty.copy()

            # Transitions from dpAll: can start a new subsequence here or extend any subsequence
            for (s, p), prod in dpAll.items():
                new_prod = prod * num
                if new_prod <= limit:
                    # compute new alternating sum
                    s2 = s + num if p == 0 else s - num
                    p2 = 1 - p
                    # update dpAll
                    if new_prod > newAll.get((s2, p2), -1):
                        newAll[(s2, p2)] = new_prod
                    # update dpNonEmpty (this subsequence is now non-empty)
                    if new_prod > newNe.get((s2, p2), -1):
                        newNe[(s2, p2)] = new_prod

            # Transitions from dpNonEmpty: extend only already-non-empty subsequences
            for (s, p), prod in dpNonEmpty.items():
                new_prod = prod * num
                if new_prod <= limit:
                    s2 = s + num if p == 0 else s - num
                    p2 = 1 - p
                    # update dpAll
                    if new_prod > newAll.get((s2, p2), -1):
                        newAll[(s2, p2)] = new_prod
                    # update dpNonEmpty
                    if new_prod > newNe.get((s2, p2), -1):
                        newNe[(s2, p2)] = new_prod

            dpAll = newAll
            dpNonEmpty = newNe

        # Among non-empty subsequences, pick the max product for alternating sum == k
        ans = -1
        for p in (0, 1):
            val = dpNonEmpty.get((k, p))
            if val is not None and val > ans:
                ans = val
        return ans