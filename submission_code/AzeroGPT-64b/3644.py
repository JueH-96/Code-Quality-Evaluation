from typing import List
from sortedcontainers import SortedList

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        result = float("inf")
        sl = SortedList()
        for i in range(l, r + 1):
            for j in range(i, n + 1):
                current_sum = prefix[j] - prefix[j - i]
                if current_sum > 0:
                    if i > l:
                        possible_match = sl.bisect_left(prefix[j] - prefix[j - i])
                        if possible_match - i != 0 and possible_match - i < j and sl[possible_match - i] > prefix[j - i]:
                            result = min(result, current_sum)
                    result = min(result, current_sum)
                if i < r:
                    sl.add(prefix[j] - prefix[j - (i + 1)])

            if i < r:
                sl.clear()

        return -1 if result == float("inf") else result