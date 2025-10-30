from collections import Counter
from math import log2
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = Counter(nums)
        ans = 0
        for num in sorted(count):
            if num == 1:
                ans = max(ans, count[num] - 1 if count[num] % 2 == 0 else count[num])
                continue
            curr = num
            curr_ans = 0
            while curr in count and count[curr] >= 2:
                curr_ans += 2
                if log2(curr) % 1 != 0:
                    break
                curr = curr**2
            if curr in count:
                curr_ans += 1
            else:
                curr_ans -= 1
            ans = max(ans, curr_ans)
        return ans