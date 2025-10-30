from typing import List
import heapq

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Check if it's possible to mark all indices by time s
        def can_mark_by(s: int) -> bool:
            # pos[i] = last time (1-based) <= s when we can mark index i+1
            pos = [0] * n
            for t in range(s):
                idx = changeIndices[t] - 1
                pos[idx] = t + 1
            # Every index must appear at least once in the first s seconds
            for i in range(n):
                if pos[i] == 0:
                    return False
            
            # Mark those marking times as blocked for decrements
            blocked = [False] * (s + 1)
            for i in range(n):
                blocked[pos[i]] = True
            
            # Build the list of decrement‐tasks, each with its deadline
            # Deadline for index i is pos[i] - 1 (we must finish its nums[i] decrements before its mark)
            tasks = []
            for i in range(n):
                cnt = nums[i]
                if cnt == 0:
                    continue
                deadline = pos[i] - 1
                # If we need more decrements than time before its marking, impossible
                if cnt > deadline:
                    return False
                # Add 'cnt' tasks each with this deadline
                tasks.extend([deadline] * cnt)
            
            # Use an EDF‐style check: always do the task with the smallest deadline first
            heapq.heapify(tasks)
            for t in range(1, s + 1):
                if blocked[t]:
                    # this second is used for marking, cannot decrement
                    continue
                if tasks:
                    d = tasks[0]
                    # if the earliest deadline is already passed, fail
                    if d < t:
                        return False
                    heapq.heappop(tasks)
                # else: no pending decrement tasks, idle or do nothing
            
            # if we've scheduled all decrement tasks, success
            return not tasks
        
        # Binary search on the earliest second s in [1..m]
        lo, hi = 1, m + 1
        while lo < hi:
            mid = (lo + hi) // 2
            if can_mark_by(mid):
                hi = mid
            else:
                lo = mid + 1
        
        # Check final candidate
        if lo <= m and can_mark_by(lo):
            return lo
        return -1