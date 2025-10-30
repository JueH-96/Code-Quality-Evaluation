from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)

        # ----------------------------------------------------------
        # helper: is it possible to finish everything within T secs
        # ----------------------------------------------------------
        def feasible(T: int) -> bool:
            # last appearance (≤ T) for every index (0-based). -1 means not present
            last = [-1] * n
            for sec in range(T):
                idx = changeIndices[sec] - 1          # convert to 0-based
                last[idx] = sec + 1                   # store 1-based second number

            # every index must appear at least once in the first T seconds
            if -1 in last:
                return False

            # prepare (deadline, decrements_needed) pairs and sort by deadline
            tasks = sorted((last[i], nums[i]) for i in range(n))

            total_decrements = 0                     # decrements already scheduled
            for cnt, (deadline, need) in enumerate(tasks, start=1):
                total_decrements += need
                # seconds before the deadline that are free for decrements
                free_slots = deadline - cnt          # (deadline-1) seconds minus (cnt-1) earlier marks
                if total_decrements > free_slots:
                    return False
            return True

        # -----------------------------
        # binary search on the answer
        # -----------------------------
        left, right, answer = 1, m, -1
        while left <= right:
            mid = (left + right) // 2
            if feasible(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        return answer