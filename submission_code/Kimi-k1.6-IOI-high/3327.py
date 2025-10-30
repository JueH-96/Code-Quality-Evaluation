import bisect
from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = []
        n = len(nums)
        for i in range(n):
            if nums[i] == 1:
                ones.append(i)
        candidates = set()
        
        # Add all ones and their adjacent positions
        for o in ones:
            for da in (-1, 0, 1):
                a = o + da
                if 0 <= a < n:
                    candidates.add(a)
        
        # Add edges
        candidates.add(0)
        candidates.add(n - 1)
        candidates = list(candidates)
        
        min_moves = float('inf')
        no_existing = len(ones) == 0
        
        for a in candidates:
            initial_free = 1 if a in ones else 0
            rem = k - initial_free
            
            if rem <= 0:
                current_sum = 0
                min_moves = min(min_moves, current_sum)
                continue
            
            other_ones = [o for o in ones if o != a]
            s_dist = len(other_ones)
            if s_dist == 0:
                current_sum = 2 * rem
                min_moves = min(min_moves, current_sum)
                continue
            
            distances = [abs(a - o) for o in other_ones]
            distances.sort()
            
            prefix_sums = [0] * (s_dist + 1)
            for i in range(s_dist):
                prefix_sums[i + 1] = prefix_sums[i] + distances[i]
            
            t_min = max(0, rem - maxChanges)
            t_max = min(rem, s_dist)
            
            m = bisect.bisect_right(distances, 2)
            possible_t = {t_min, t_max, m}
            possible_t = list(possible_t)
            valid_t = [
                t for t in possible_t 
                if t_min <= t <= t_max and 0 <= t <= s_dist
            ]
            
            if not valid_t:
                valid_t = [t_min]
            
            current_min = float('inf')
            for t in valid_t:
                sum_d = prefix_sums[t] if t <= s_dist else prefix_sums[s_dist]
                total = sum_d + 2 * (rem - t)
                if total < current_min:
                    current_min = total
            
            min_moves = min(min_moves, current_min)
        
        if no_existing:
            min_moves = min(min_moves, 2 * k)
        
        return min_moves if min_moves != float('inf') else 0