import bisect
from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        index_map = [[] for _ in range(1001)]
        for i in range(n):
            index_map[nums[i]].append(i)
        
        values_set = set(nums)
        ans = 0
        for q in range(2, n - 4):  # q from 2 to n-5 inclusive
            for r in range(q + 2, n - 2):  # r from q+2 to n-3 inclusive
                B = nums[q]
                C = nums[r]
                for A in values_set:
                    if (A * C) % B == 0:
                        D = (A * C) // B
                        if 1 <= D <= 1000 and D in values_set:
                            lst_p = index_map[A]
                            num_p = bisect.bisect_right(lst_p, q - 2)
                            lst_s = index_map[D]
                            num_s = len(lst_s) - bisect.bisect_left(lst_s, r + 2)
                            ans += num_p * num_s
        return ans