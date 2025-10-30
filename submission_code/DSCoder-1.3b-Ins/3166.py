from typing import List
import heapq

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = [0] * (10**9 + 1)
        for num in nums:
            count[num] += 1

        groups = []
        for num, cnt in enumerate(count):
            if cnt:
                heapq.heappush(groups, (cnt, num))

        valid_groups = 0
        while groups:
            cnt, num = heapq.heappop(groups)
            if cnt == 1:
                valid_groups += 1
            else:
                cnt -= 1
                if cnt > 0:
                    heapq.heappush(groups, (cnt, num))

        return valid_groups