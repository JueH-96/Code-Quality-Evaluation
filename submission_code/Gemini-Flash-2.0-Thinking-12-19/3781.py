import sys
from typing import List
from functools import lru_cache

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        n = len(points)

        # Precompute all pairwise Manhattan distances
        # Use a 2D list for simplicity
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist[i][j] = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                dist[j][i] = dist[i][j] # Distance is symmetric

        # Check if it's possible to select k points with minimum distance at least D
        # This is equivalent to checking if the graph where edges represent distance < D
        # has an independent set of size k.
        # We use recursive backtracking with memoization.
        # State: (mask of available points, number of points needed)
        # The mask is an integer where the i-th bit is 1 if point i is available.
        # count_needed is the number of additional points we need to select.
        @lru_cache(None) # Memoize the function calls to avoid recomputing states
        def can_select(mask: int, count_needed: int) -> bool:
            # Base case 1: If we have successfully selected k points (i.e., count_needed reached 0)
            if count_needed == 0:
                return True

            # Optimization 1: If there are fewer points available (set bits in mask) than needed, it's impossible.
            num_available = bin(mask).count('1')
            if num_available < count_needed:
                return False

            # Base case 2: If no points are available but we still need points (this is covered by Opt 1 if count_needed > 0)
            # if mask == 0:
            #      return False # Redundant if count_needed > 0

            # Branching step: Pick the smallest index `i` that is still in the mask.
            # We must either include point `i` in our selection or exclude it.
            i = 0
            # Find the index of the first set bit (smallest available index)
            # This loop is safe because mask > 0 (due to Opt 1 and initial call) guarantees at least one bit is set.
            while i < n and (mask >> i) & 1 == 0:
                 i += 1

            # Option 1: Try selecting point `i`.
            # If we select `i`, we need `count_needed - 1` more points.
            # Any point `j` that has distance less than D from `i` cannot be selected
            # along with `i`. We remove `i` and all its "close" neighbors (distance < D)
            # that are currently available in the mask.

            next_mask_select = mask & ~(1 << i) # Remove i from future consideration

            # Determine which points `j` currently in the mask are too close to `i` (distance < D).
            # These points must be excluded from the set of available points for the next recursive call.
            neighbor_mask = 0
            for j in range(n):
                # Check if point j is available in the current mask and is different from i
                if ((mask >> j) & 1) and j != i:
                     if dist[i][j] < D:
                         neighbor_mask |= (1 << j) # Mark j as a neighbor to be removed

            # Recursively call can_select with the updated mask (excluding i and its neighbors)
            # and needing one fewer point.
            if can_select(next_mask_select & ~neighbor_mask, count_needed - 1):
                 return True

            # Option 2: Try not selecting point `i`.
            # If we don't select `i`, we still need `count_needed` points, but point `i` is no longer available.
            # We create a new mask `next_mask_skip` by simply removing `i` from the current mask.

            next_mask_skip = mask & ~(1 << i) # Remove i from future consideration

            # Recursively call can_select with the updated mask (excluding only i)
            # and needing the same number of points.
            if can_select(next_mask_skip, count_needed):
                 return True

            # If neither selecting nor skipping point `i` leads to a solution, then it's impossible
            # to select `count_needed` points from the current `mask` under consideration.
            return False

        # Binary search for the maximum possible minimum distance D.
        # The range of possible minimum distances is [0, 2*side].
        # The maximum possible Manhattan distance between any two points on the boundary is 2*side
        # (e.g., between (0,0) and (side,side)).
        # We search in the range [low, high).
        low = 0
        high = 2 * side + 1 # Search space is [low, high). High is exclusive.
        ans = 0 # Stores the maximum D for which can_select is True.

        while low < high:
            mid = low + (high - low) // 2
            D = mid

            # Clear the memoization cache for the recursive function `can_select`.
            # The results of can_select depend on the value of D, so the cache
            # must be cleared for each iteration of the binary search.
            can_select.cache_clear()

            # Check if it's possible to select k points with the current minimum distance D.
            # Start the recursion with all points available (mask = (1<<n)-1) and needing k points.
            if can_select((1 << n) - 1, k):
                # If possible to select k points with min distance D, then D is achievable.
                # We store this D as a potential answer and try to achieve a larger distance.
                ans = D
                low = mid + 1
            else:
                # If not possible to select k points with min distance D, then D is too large.
                # We need to try a smaller distance.
                high = mid

        # After the binary search loop finishes, `ans` holds the maximum value of D
        # for which `can_select((1 << n) - 1, k)` returned True.
        return ans