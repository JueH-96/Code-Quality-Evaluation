from typing import List
from bisect import bisect_left
from collections import defaultdict

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_pairs = n // 2
        freq = defaultdict(int)
        max_ab_list = []
        
        for i in range(total_pairs):
            a = nums[i]
            b = nums[n - i - 1]
            original_X = abs(a - b)
            freq[original_X] += 1
            
            max_a = max(a, k - a)
            max_b = max(b, k - b)
            max_ab = max(max_a, max_b)
            max_ab_list.append(max_ab)
        
        max_ab_list.sort()
        
        min_sum = float('inf')
        for X in range(0, k + 1):
            count_equals = freq.get(X, 0)
            count_not_in_S = bisect_left(max_ab_list, X)
            current_sum = (total_pairs - count_equals) + count_not_in_S
            if current_sum < min_sum:
                min_sum = current_sum
        
        return min_sum