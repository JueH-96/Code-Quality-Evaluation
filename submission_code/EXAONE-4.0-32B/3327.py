import bisect
from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = [i for i in range(n) if nums[i] == 1]
        total_ones = len(ones)
        
        if total_ones == 0:
            return 2 * k
        
        candidate_positions = set()
        for i in ones:
            candidate_positions.add(i)
            if i - 1 >= 0:
                candidate_positions.add(i - 1)
            if i + 1 < n:
                candidate_positions.add(i + 1)
        
        INF = 10**18
        ans = INF
        
        for pos in candidate_positions:
            free = 1 if nums[pos] == 1 else 0
            rem = k - free
            if rem < 0:
                continue
            if rem == 0:
                ans = min(ans, 0)
                continue
            
            if nums[pos] == 1:
                left_ones = [x for x in ones if x < pos]
                right_ones = [x for x in ones if x > pos]
            else:
                left_ones = [x for x in ones if x < pos]
                right_ones = [x for x in ones if x > pos]
            
            left_dists = [pos - x for x in reversed(left_ones)]
            right_dists = [x - pos for x in right_ones]
            
            prefix_left = [0]
            for d in left_dists:
                prefix_left.append(prefix_left[-1] + d)
            prefix_right = [0]
            for d in right_dists:
                prefix_right.append(prefix_right[-1] + d)
            
            r = max(0, rem - maxChanges)
            if r == 0:
                cost = 2 * rem
                ans = min(ans, cost)
            else:
                low, high = 0, n
                while low < high:
                    mid = (low + high) // 2
                    cnt_left = bisect.bisect_right(left_dists, mid)
                    cnt_right = bisect.bisect_right(right_dists, mid)
                    total_cnt = cnt_left + cnt_right
                    if total_cnt < r:
                        low = mid + 1
                    else:
                        high = mid
                D = low
                cnt_left = bisect.bisect_right(left_dists, D)
                cnt_right = bisect.bisect_right(right_dists, D)
                total_cnt = cnt_left + cnt_right
                sum_left = prefix_left[cnt_left]
                sum_right = prefix_right[cnt_right]
                total_sum = sum_left + sum_right
                if total_cnt > r:
                    total_sum -= D * (total_cnt - r)
                cost = 2 * (rem - r) + total_sum
                ans = min(ans, cost)
        
        return ans