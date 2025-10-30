import bisect
from typing import List

class Solution:
    def generate_palindromes(self):
        pals = []
        for d in range(1, 10):
            start = 10**((d-1)//2)
            end = 10**((d+1)//2) - 1
            for root in range(start, end + 1):
                s = str(root)
                if d % 2 == 0:
                    full_s = s + s[::-1]
                else:
                    full_s = s + s[:-1][::-1]
                p = int(full_s)
                if p < 10**9:
                    pals.append(p)
        return pals

    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        b = sorted(nums)
        min_val, max_val = b[0], b[-1]
        
        prefix = [0] * n
        prefix[0] = b[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] + b[i]
        total_sum = prefix[-1]
        
        pals_list = self.generate_palindromes()
        
        left_index = bisect.bisect_left(pals_list, min_val)
        right_index = bisect.bisect_right(pals_list, max_val) - 1
        
        if left_index <= right_index:
            candidates = pals_list[left_index:right_index+1]
        else:
            candidates = []
            if left_index > 0:
                candidates.append(pals_list[left_index-1])
            if left_index < len(pals_list):
                candidates.append(pals_list[left_index])
        
        min_cost = float('inf')
        for y in candidates:
            pos = bisect.bisect_right(b, y)
            idx = pos - 1
            if idx < 0:
                count_left = 0
                sum_left = 0
            else:
                count_left = idx + 1
                sum_left = prefix[idx]
            count_right = n - count_left
            sum_right = total_sum - sum_left
            left_cost = count_left * y - sum_left
            right_cost = sum_right - count_right * y
            total_cost = left_cost + right_cost
            if total_cost < min_cost:
                min_cost = total_cost
                
        return min_cost