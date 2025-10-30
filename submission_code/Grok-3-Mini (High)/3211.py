from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        def can_partition(L: int) -> bool:
            b_prev = 0
            current_min_sum = 0
            for part_num in range(1, L + 1):
                if part_num < L:
                    target = prefix[b_prev] + current_min_sum
                    start_idx = b_prev + 1
                    low = start_idx
                    high = n
                    while low <= high:
                        mid_idx = (low + high) // 2
                        if prefix[mid_idx] >= target:
                            high = mid_idx - 1
                        else:
                            low = mid_idx + 1
                    if low > n or prefix[low] < target:
                        return False
                    b_curr = low
                    s_curr = prefix[b_curr] - prefix[b_prev]
                    current_min_sum = s_curr
                    b_prev = b_curr
                else:  # last part
                    if n <= b_prev:
                        return False
                    sum_last = prefix[n] - prefix[b_prev]
                    if sum_last >= current_min_sum:
                        return True
                    else:
                        return False
        
        left, right = 1, n
        result = 1  # L=1 is always possible
        while left <= right:
            mid = (left + right) // 2
            if can_partition(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result