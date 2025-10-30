import bisect
from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = [i for i, val in enumerate(nums) if val == 1]
        S = len(ones)
        
        def get_m_closest_sum(other_ones, pos, m):
            n = len(other_ones)
            if m == 0:
                return 0
            # Find insert position
            left, right = 0, n - 1
            insert_pos = 0
            while left <= right:
                mid = (left + right) // 2
                if other_ones[mid] < pos:
                    insert_pos = mid + 1
                    left = mid + 1
                else:
                    right = mid - 1
            # Now, insert_pos is the first index >= pos
            i = insert_pos - 1  # left pointer
            j = insert_pos      # right pointer
            res = []
            while i >= 0 and j < n and len(res) < m:
                dist_i = pos - other_ones[i]
                dist_j = other_ones[j] - pos
                if dist_i < dist_j:
                    res.append(other_ones[i])
                    i -= 1
                else:
                    res.append(other_ones[j])
                    j += 1
            # Take remaining from i or j
            while len(res) < m and i >= 0:
                res.append(other_ones[i])
                i -= 1
            while len(res) < m and j < n:
                res.append(other_ones[j])
                j += 1
            # Calculate sum of distances
            total = 0
            for x in res:
                total += abs(x - pos)
            return total
        
        min_case1 = float('inf')
        if S > 0 and k >= 1:
            req_case1 = k - 1
            for pos in ones:
                # Find the index of pos in ones
                idx = bisect.bisect_left(ones, pos)
                if idx < len(ones) and ones[idx] == pos:
                    # Create other_ones by removing pos
                    other_ones = ones[:idx] + ones[idx+1:]
                else:
                    # Should not happen since pos is in ones
                    other_ones = ones.copy()
                m_start = max(0, req_case1 - maxChanges)
                m_end = min(req_case1, len(other_ones))
                if m_start > m_end:
                    continue
                current_min = float('inf')
                for m in range(m_start, m_end + 1):
                    sum_dist = get_m_closest_sum(other_ones, pos, m)
                    created = req_case1 - m
                    if created < 0 or created > maxChanges:
                        continue
                    total = sum_dist + 2 * created
                    if total < current_min:
                        current_min = total
                if current_min < min_case1:
                    min_case1 = current_min
        
        def compute_min_req(req):
            if S == 0:
                if req <= maxChanges:
                    return req * 2
                else:
                    return float('inf')
            prefix = [0] * (S + 1)
            for i in range(S):
                prefix[i+1] = prefix[i] + ones[i]
            min_total = float('inf')
            m_start = max(0, req - maxChanges)
            m_end = min(req, S)
            for m in range(m_start, m_end + 1):
                if m == 0:
                    if req <= maxChanges:
                        candidate = req * 2
                        if candidate < min_total:
                            min_total = candidate
                    continue
                current_min = float('inf')
                for l in range(0, S - m + 1):
                    r = l + m - 1
                    mid = l + (m - 1) // 2
                    mid_val = ones[mid]
                    # Calculate left_sum
                    left_sum = (mid - l + 1) * mid_val - (prefix[mid + 1] - prefix[l])
                    # Calculate right_sum
                    right_sum = (prefix[r + 1] - prefix[mid + 1]) - (r - mid) * mid_val
                    total = left_sum + right_sum
                    if total < current_min:
                        current_min = total
                created = req - m
                if created < 0 or created > maxChanges:
                    continue
                current_total = current_min + 2 * created
                if current_total < min_total:
                    min_total = current_total
            # Check all created
            if req <= maxChanges:
                candidate = req * 2
                if candidate < min_total:
                    min_total = candidate
            return min_total
        
        min_case2 = compute_min_req(k)
        
        result = min_case2
        if S > 0 and k >= 1:
            result = min(result, min_case1)
        return result