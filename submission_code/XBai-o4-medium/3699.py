from typing import List
from collections import defaultdict

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        # Precompute right_freq: for each i, right_freq[i] is a dictionary of counts for nums[i+2 ... n-1]
        right_freq = [dict() for _ in range(n)]
        for i in range(n-1, -1, -1):
            if i + 2 >= n:
                right_freq[i] = dict()
            else:
                right_freq[i] = right_freq[i+1].copy()
                val = nums[i+2]
                right_freq[i][val] = right_freq[i].get(val, 0) + 1
        
        # Precompute left_freq: for each q, left_freq[q] is a dictionary of counts for nums[0 ... q-2]
        left_freq = [dict() for _ in range(n)]
        current_left = dict()
        for q in range(n):
            left_freq[q] = current_left.copy()
            if q - 1 >= 0:
                val = nums[q-1]
                current_left[val] = current_left.get(val, 0) + 1
        
        total = 0
        # Iterate over all valid q and r
        for q in range(n):
            for r in range(q + 2, n):
                if r + 2 >= n:
                    continue
                B = nums[q]
                C = nums[r]
                current_left_dict = left_freq[q]
                for A in current_left_dict:
                    product = A * C
                    if product % B == 0:
                        D = product // B
                        count_s = right_freq[r].get(D, 0)
                        total += current_left_dict[A] * count_s
        return total