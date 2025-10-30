from collections import defaultdict
from typing import List

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # Store pairs grouped by the smaller number, also storing the original index.
        # Ensure pairs are always (min(a, b), max(a, b))
        pairs_at_u = defaultdict(list)
        for i, pair in enumerate(conflictingPairs):
            u, v = sorted(pair)
            pairs_at_u[u].append((v, i))

        # Sort pairs at each starting index by the ending number v
        for u in pairs_at_u:
            pairs_at_u[u].sort()

        # DP arrays for minimum v, count of minimum, second minimum v, and index of unique minimum owner
        # Arrays are 1-indexed for easier mapping with problem statement (1 to n)
        # Size N+2 to handle indices 1 to N and N+1 padding
        min1_arr = [n + 1] * (n + 2)
        min1_count = [0] * (n + 2)
        min2_arr = [n + 1] * (n + 2)
        min1_owner_idx = [-1] * (n + 2) # Stores original index of the unique pair that sets min1 if count is 1

        # Initialize padding state for index n+1
        min1_arr[n+1] = n+1
        min1_count[n+1] = 0
        min2_arr[n+1] = n+1
        min1_owner_idx[n+1] = -1


        # Iterate backwards from N down to 1
        for l in range(n, 0, -1):
            # State from the next index l+1
            m1_next = min1_arr[l + 1]
            c1_next = min1_count[l + 1]
            m2_next = min2_arr[l + 1]
            owner_next = min1_owner_idx[l + 1]

            # Pairs starting exactly at l, sorted by v
            current_pairs = pairs_at_u.get(l, []) # list of (v, i)

            # Find min1, count1, owner from current_pairs
            current_min_v = n + 1
            current_min_count = 0
            current_min_owner = -1 # Index of the unique pair at l that achieves current_min_v

            if current_pairs:
                current_min_v = current_pairs[0][0]
                for j in range(len(current_pairs)):
                    if current_pairs[j][0] == current_min_v:
                        current_min_count += 1
                        # The potential unique owner at this level l is the first pair with the min value
                        if current_min_count == 1:
                             current_min_owner = current_pairs[j][1]
                    else:
                         break # Since sorted

            # Determine the new min1, count1, min2, owner for l
            new_m1 = n + 1
            new_c1 = 0
            new_m2 = n + 1
            new_owner = -1 # Determined based on final count


            second_v_from_current = n + 1
            if current_min_count > 0 and len(current_pairs) > current_min_count:
                second_v_from_current = current_pairs[current_min_count][0]


            if current_min_v < m1_next:
                new_m1 = current_min_v
                new_c1 = current_min_count
                new_m2 = min(m1_next, m2_next, second_v_from_current)

            elif current_min_v == m1_next:
                new_m1 = m1_next
                new_c1 = c1_next + current_min_count
                new_m2 = min(m2_next, second_v_from_current)

            else: # m1_next < current_min_v
                new_m1 = m1_next
                new_c1 = c1_next
                new_m2 = min(m2_next, current_min_v)

            # Determine the overall unique owner index for the new_m1 if new_c1 is 1
            if new_c1 == 1:
                 # The unique source must come from either the current level (l) or the next level (l+1)
                 if new_m1 == current_min_v and current_min_count == 1:
                     # The unique minimum came from a unique pair at level l
                     new_owner = current_min_owner
                 elif new_m1 == m1_next and c1_next == 1:
                     # The unique minimum came from a unique source at level l+1
                     new_owner = owner_next
                 # If new_m1 is set by both current_min_v and m1_next simultaneously, and one/both are unique sources?
                 # This happens in the elif current_min_v == m1_next case, but then new_c1 >= 2.
                 # So, the logic `if new_c1 == 1: ...` correctly identifies the unique source if it exists.


            min1_arr[l] = new_m1
            min1_count[l] = new_c1
            min2_arr[l] = new_m2
            min1_owner_idx[l] = new_owner


        # Calculate BaseValid
        BaseValid = 0
        for l in range(1, n + 1):
          # Number of valid subarrays starting at l is max(0, max_r - l + 1)
          # max_r = min1_arr[l] - 1
          # Number of r is (min1_arr[l] - 1) - l + 1 = min1_arr[l] - l
          # Ensure non-negative count
          BaseValid += max(0, min1_arr[l] - l)


        # Calculate increase for each removed pair
        increases = [0] * len(conflictingPairs)

        for l in range(1, n + 1):
            # If the minimum constraint at l is unique (count is 1)
            if min1_count[l] == 1:
                owner_idx = min1_owner_idx[l]
                # Ensure the owner index is valid (came from an original pair)
                if owner_idx != -1:
                    # This unique minimum constraint min1_arr[l] is imposed by
                    # the original pair conflictingPairs[owner_idx].
                    # If we remove this specific pair, the constraint at l changes to min2_arr[l].
                    # The number of valid subarrays starting at l changes from max(0, min1_arr[l] - l)
                    # to max(0, min2_arr[l] - l).
                    increase_l = max(0, min2_arr[l] - l) - max(0, min1_arr[l] - l)
                    increases[owner_idx] += increase_l

        # The maximum number of valid subarrays is BaseValid + the maximum possible increase
        # Use max(increases + [0]) to handle potential edge cases or ensure correct behavior
        # if increases was unexpectedly empty (though constraints prevent this).
        max_increase = max(increases) if increases else 0

        return BaseValid + max_increase