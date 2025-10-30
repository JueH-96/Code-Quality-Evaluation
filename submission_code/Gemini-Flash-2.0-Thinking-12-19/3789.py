from typing import List
import bisect

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # Preprocess conflictingPairs to store min/max consistently
        processed_pairs = []
        u_to_vs = {}
        pairs_ending_at = {}
        distinct_v_values = set()

        for pair in conflictingPairs:
            u, v = min(pair), max(pair)
            if u >= v: # Ensure u < v
                continue
            processed_pairs.append((u, v))
            
            if u not in u_to_vs:
                u_to_vs[u] = []
            u_to_vs[u].append(v)
            
            if v not in pairs_ending_at:
                pairs_ending_at[v] = []
            pairs_ending_at[v].append(u)
            distinct_v_values.add(v)

        # Sort v values for each u
        for u in u_to_vs:
            u_to_vs[u].sort()

        # Compute max1[e] and max2[e] for e from 0 to n
        # max1[e] = max u | exists [u,v] in C_all, u<v, v <= e
        # max2[e] = second max u | exists [u,v] in C_all, u<v, v <= e
        # Initialize with 0s
        max1 = [0] * (n + 1)
        max2 = [0] * (n + 1)

        for e in range(1, n + 1):
            current_max1 = max1[e-1]
            current_max2 = max2[e-1]

            if e in pairs_ending_at:
                us_at_e = sorted(pairs_ending_at[e], reverse=True) # Process larger u first
                for u in us_at_e:
                    # u < v = e condition is implicitly handled as u comes from pairs_ending_at[e]
                    if u > current_max1:
                        current_max2 = current_max1
                        current_max1 = u
                    elif u > current_max2:
                        current_max2 = u

            max1[e] = current_max1
            max2[e] = current_max2

        # Compute total valid subarrays if no pair was removed
        # Sum_{e=1..n} max(0, e - max_forbidden_start_all[e])
        total_valid_all = 0
        for e in range(1, n + 1):
            total_valid_all += max(0, e - max1[e])

        # Find maximum CountUniqueInvalid(p) over all pairs p
        max_count_unique_invalid = 0

        # Base critical points for e intervals are distinct v values from conflicting pairs + 0 and n
        base_critical_e_points = sorted(list(distinct_v_values | {0, n}))
        
        # Iterate through each pair to calculate its CountUniqueInvalid
        for u0, v0 in processed_pairs:
            count_unique_invalid_pair = 0

            # Get properties for this pair [u0, v0]
            # Guaranteed u0 < v0 and u0 is in u_to_vs
            u0_vs = u_to_vs[u0] 
            nu = len(u0_vs)
            v_first = u0_vs[0]
            v_second = u0_vs[1] if nu >= 2 else float('inf')

            # The sum for CountUniqueInvalid([u0, v0]) is over e from v0 to n.
            # The condition R_u0_v0(e) is true iff v_first == v0 and (nu == 1 or v_second > e).
            # This condition R_u0_v0(e) changes its truth value only at e = v_second - 1 (if v_first == v0 and nu >= 2).

            # Define the ranges for e where R_u0_v0(e) has a constant truth value within [v0, n]
            sum_ranges = []
            if v_first == v0:
                if nu == 1:
                    # R is always true for e >= v0
                    sum_ranges.append((v0, n, True))
                else: # nu >= 2
                    # R is true for e in [v0, v_second - 1], false for e in [v_second, n]
                    if v0 <= v_second - 1: # Ensure the true range is valid
                        sum_ranges.append((v0, min(n, v_second - 1), True))
                    if v_second <= n: # Ensure the false range is valid
                        sum_ranges.append((max(v0, v_second), n, False))
            else: # v_first != v0, R is always false for e >= v0
                 if v0 <= n:
                     sum_ranges.append((v0, n, False))


            # Iterate through each sum range [sum_start, sum_end] with fixed R_status
            for sum_start, sum_end, is_R_true_in_range in sum_ranges:

                # Iterate through intervals [I_start, I_end] defined by distinct v values
                # Intersect [sum_start, sum_end] with these intervals
                
                for i in range(len(base_critical_e_points) - 1):
                     I_start = base_critical_e_points[i] + 1
                     I_end = base_critical_e_points[i+1]

                     if I_start > I_end:
                         continue

                     # Overlap interval [O_start, O_end]
                     O_start = max(sum_start, I_start)
                     O_end = min(sum_end, I_end)

                     if O_start > O_end:
                         continue

                     # Get constant max1, max2 for this interval [O_start, O_end]
                     # Use the value at the end of the interval, as max1/max2 are non-decreasing
                     M1 = max1[O_end] 
                     M2 = max2[O_end] 
                     overlap_len = O_end - O_start + 1

                     # Calculate contribution from this overlap interval
                     # Sum_{e=O_start..O_end} max(0, u0 - effective_max_u(e, [u0, v0]))
                     # Based on the formula:
                     # Sum_{e, u0 > max1} (u0 - max1) + Sum_{e, u0 == max1, u0 > max2, R} (u0 - max2)

                     # Contribution from the first term: Sum_{e=O_start..O_end, u0 > M1} (u0 - M1)
                     if u0 > M1:
                          # Since max1[e] is constant M1 in this interval
                          count_unique_invalid_pair += overlap_len * (u0 - M1)
                     
                     # Contribution from the second term: Sum_{e=O_start..O_end, u0 == M1, u0 > M2, R} (u0 - M2)
                     if u0 == M1 and u0 > M2 and is_R_true_in_range:
                          # Since max1[e]=M1 and max2[e]=M2 constant in this interval, and R is true
                          count_unique_invalid_pair += overlap_len * (u0 - M2)
                
            max_count_unique_invalid = max(max_count_unique_invalid, count_unique_invalid_pair)

        return total_valid_all + max_count_unique_invalid