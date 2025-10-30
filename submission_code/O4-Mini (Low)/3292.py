from typing import List
import bisect

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        # Build for each index the list of times it appears in changeIndices
        occ = [[] for _ in range(n)]
        for t, idx in enumerate(changeIndices, start=1):
            occ[idx-1].append(t)
        # Quick check: any index with no occurrences at all => cannot ever mark it
        for i in range(n):
            if not occ[i]:
                return -1
        
        def can_by(T: int) -> bool:
            # For each index, find its chosen marking time t_i = last occurrence <= T
            tasks = []  # list of (deadline t_i, work w_i)
            total_work = 0
            for i in range(n):
                # find rightmost time <= T in occ[i]
                lst = occ[i]
                # bisect_right gives insertion point >T, so take idx-1
                j = bisect.bisect_right(lst, T) - 1
                if j < 0:
                    # no occurrence <= T, can't mark
                    return False
                t_i = lst[j]
                w_i = nums[i]
                total_work += w_i
                tasks.append((t_i, w_i))
            # Total available decrement slots is T - n (we must reserve n slots for the markings)
            if total_work > T - n:
                return False
            # Sort tasks by deadline
            tasks.sort()
            used_work = 0
            used_marks = 0
            # We iterate tasks grouped by deadline
            i = 0
            while i < n:
                t_dead = tasks[i][0]
                # accumulate all with same deadline
                while i < n and tasks[i][0] == t_dead:
                    used_work += tasks[i][1]
                    used_marks += 1
                    i += 1
                # By time t_dead, capacity for decrements = t_dead - (#marks scheduled by then)
                if used_work > t_dead - used_marks:
                    return False
            return True

        # Binary search earliest T in [1..m]
        lo, hi = 1, m
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_by(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans