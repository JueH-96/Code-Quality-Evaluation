import bisect
from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        starts = [i for i in range(n) if nums[i] == 0]
        count = 0
        for s in starts:
            for d in [1, -1]:
                arr = nums.copy()
                active = [i for i in range(n) if arr[i] != 0]
                active.sort()
                cur = s
                direction = d
                while 0 <= cur < n:
                    if arr[cur] > 0:
                        arr[cur] -= 1
                        if arr[cur] == 0:
                            idx = bisect.bisect_left(active, cur)
                            if idx < len(active) and active[idx] == cur:
                                del active[idx]
                        direction = -direction
                        cur += direction
                    else:
                        if direction == 1:
                            pos = bisect.bisect_right(active, cur)
                            if pos < len(active):
                                cur = active[pos]
                            else:
                                cur = -1
                        else:
                            pos = bisect.bisect_left(active, cur)
                            if pos > 0:
                                cur = active[pos-1]
                            else:
                                cur = -1
                        if cur == -1:
                            break
                if all(x == 0 for x in arr):
                    count += 1
        return count