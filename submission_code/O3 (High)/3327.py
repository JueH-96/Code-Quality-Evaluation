from bisect import bisect_left, bisect_right
from typing import List


class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)

        # collect indices of 1's
        ones_pos = [i for i, v in enumerate(nums) if v == 1]
        m = len(ones_pos)

        # when the array originally has no 1's we can only rely on changes
        if m == 0:
            return 2 * k          # flip (2-cost) for every required one

        # prefix sums of positions – for O(1) range-sum queries
        prefix = [0]
        for p in ones_pos:
            prefix.append(prefix[-1] + p)

        def range_count(l: int, r: int) -> int:
            """how many ones indices in [l, r]"""
            left = bisect_left(ones_pos, l)
            right = bisect_right(ones_pos, r)
            return right - left

        def range_sum(l: int, r: int) -> int:
            """sum of indices of ones in [l, r]"""
            left = bisect_left(ones_pos, l)
            right = bisect_right(ones_pos, r)
            return prefix[right] - prefix[left]

        def cnt_sum_within(center: int, radius: int):
            """
            number of 1's whose distance to `center` not larger than `radius`
            and the total sum of their distances
            """
            l = center - radius
            r = center + radius
            cnt = range_count(l, r)
            if cnt == 0:
                return 0, 0

            # split to the left / right of center to compute distance sum
            left_idx_end = bisect_right(ones_pos, center, 0, m)   # first idx > center
            left_idx_start = bisect_left(ones_pos, l)
            right_idx_start = left_idx_end
            right_idx_end = bisect_right(ones_pos, r)

            left_cnt = left_idx_end - left_idx_start
            right_cnt = right_idx_end - right_idx_start

            left_sum = prefix[left_idx_end] - prefix[left_idx_start]
            right_sum = prefix[right_idx_end] - prefix[right_idx_start]

            dist_sum = left_cnt * center - left_sum + right_sum - right_cnt * center
            return cnt, dist_sum

        INF = 10 ** 20
        answer = INF

        # helper arrays for O(1) look-ups of nums around position i
        nums_ext = nums  # alias for brevity

        for pos in range(n):
            # ------- gather very close (distance<=2) information ----------
            free_pick = 1 if nums_ext[pos] == 1 else 0

            adj1_cnt = 0
            if pos - 1 >= 0 and nums_ext[pos - 1]:
                adj1_cnt += 1
            if pos + 1 < n and nums_ext[pos + 1]:
                adj1_cnt += 1

            dist2_cnt = 0
            if pos - 2 >= 0 and nums_ext[pos - 2]:
                dist2_cnt += 1
            if pos + 2 < n and nums_ext[pos + 2]:
                dist2_cnt += 1

            near_cnt = free_pick + adj1_cnt + dist2_cnt

            # also need sum of their distances for later subtraction
            near_dist_sum = (adj1_cnt * 1) + (dist2_cnt * 2)        # free pick contributes 0

            # ---------------- pick using costs 0 / 1 / 2 -------------------
            remaining = k
            cost = 0

            if free_pick and remaining:
                remaining -= 1      # cost 0

            # distance 1 ones
            take1 = min(adj1_cnt, remaining)
            cost += take1 * 1
            remaining -= take1

            # all sources with cost 2 : distance-2 ones + flips (maxChanges)
            twos_available = dist2_cnt + maxChanges
            take2 = min(twos_available, remaining)
            cost += take2 * 2
            remaining -= take2

            if remaining == 0:
                answer = min(answer, cost)
                continue           # no need for farther ones

            # ---------------- need `remaining` farther (distance>=3) ones ----------------
            # binary search minimal radius such that enough far ones exist
            low, high = 3, n        # distances never exceed n-1
            while low < high:
                mid = (low + high) // 2
                cnt_total = range_count(pos - mid, pos + mid)
                far_cnt = cnt_total - near_cnt
                if far_cnt >= remaining:
                    high = mid
                else:
                    low = mid + 1
            D = low

            # data for radius D and D-1
            cnt_D, sum_D = cnt_sum_within(pos, D)
            far_cnt_D = cnt_D - near_cnt
            sum_far_D = sum_D - near_dist_sum

            if D - 1 >= 3:
                cnt_Dm1, sum_Dm1 = cnt_sum_within(pos, D - 1)
                far_cnt_Dm1 = cnt_Dm1 - near_cnt
                sum_far_Dm1 = sum_Dm1 - near_dist_sum
            else:
                far_cnt_Dm1 = 0
                sum_far_Dm1 = 0

            take_from_D = remaining - far_cnt_Dm1      # how many with exact distance D we still need
            far_cost = sum_far_Dm1 + take_from_D * D

            total_cost_here = cost + far_cost
            answer = min(answer, total_cost_here)

        return answer