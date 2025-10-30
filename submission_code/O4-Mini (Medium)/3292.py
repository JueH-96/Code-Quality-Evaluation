from typing import List
import bisect

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        # Build positions list: for each index i (0-based), list of seconds where it appears in changeIndices
        pos = [[] for _ in range(n)]
        for sec, ci in enumerate(changeIndices, start=1):
            pos[ci-1].append(sec)
        
        # Check feasibility by time T
        def can_mark_by(T: int) -> bool:
            # For each i, find the latest marking second t_i <= T
            tasks = []  # list of (deadline, processing_time)
            for i in range(n):
                lst = pos[i]
                # find rightmost position <= T
                idx = bisect.bisect_right(lst, T) - 1
                if idx < 0:
                    # no opportunity to mark index i by time T
                    return False
                t_i = lst[idx]
                p_i = nums[i]
                tasks.append((t_i, p_i))
            # sort tasks by their marking time (deadline for decrements is t_i - 1,
            # but our combined condition uses t_i directly)
            tasks.sort(key=lambda x: x[0])
            
            sum_p = 0
            count = 0
            # EDF condition: for each prefix up to deadline t,
            # sum of p_i + count_of_tasks <= t
            for t_i, p_i in tasks:
                sum_p += p_i
                count += 1
                if sum_p + count > t_i:
                    return False
            return True
        
        # Binary search the minimal T in [1..m]
        lo, hi = 1, m
        answer = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_mark_by(mid):
                answer = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        return answer