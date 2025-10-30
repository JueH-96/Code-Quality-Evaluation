class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # We want to choose a single pair (x, y) to fill in all missing (-1) entries
        # in such a way that the maximum absolute difference between adjacent elements
        # is minimized.  We can also independently decide which missing gets x or y.
        #
        # Key idea:
        # 1. Let base_diff be the maximum absolute difference between adjacent known (non -1) elements in nums.
        # 2. Collect all known values that are next to a missing position in a list "neighbors".
        # 3. If "neighbors" is empty, it means either all elements are known without touching -1
        #    or all are -1. In the "all are -1" case, we can fill them with the same number
        #    resulting in 0. If there are known adjacent pairs, those define base_diff already.
        # 4. Otherwise, let m = min(neighbors), M = max(neighbors). The best strategy to fill
        #    appears when we pick x and y around the midpoint. The maximum adjacency difference
        #    from a known neighbor to its chosen fill is at most ceil((M - m)/2).
        # 5. The final answer is max(base_diff, half_gap) where half_gap = ceil((M - m)/2).
        #
        # Steps:
        #   - Compute base_diff from known-known pairs.
        #   - Collect neighbor values from known-missing pairs.
        #   - If neighbors is not empty, half_gap = (M - m + 1)//2. Answer = max(base_diff, half_gap).
        #   - If neighbors is empty but there is at least some known-known adjacency, answer = base_diff.
        #   - If neighbors is empty and no known-known adjacency, everything is -1, answer = 0.

        n = len(nums)
        base_diff = 0
        neighbors = []

        # Calculate base_diff (max adjacent difference among known elements)
        # and gather neighbors for known-missing edges
        for i in range(n - 1):
            a, b = nums[i], nums[i + 1]
            if a != -1 and b != -1:
                base_diff = max(base_diff, abs(a - b))
            elif a != -1 and b == -1:
                neighbors.append(a)
            elif a == -1 and b != -1:
                neighbors.append(b)

        # If no neighbors, either everything is -1 or we only have known-known with no missing
        if not neighbors:
            # If there's any known-known adjacency, base_diff is already computed
            # Otherwise, everything is -1 => we can fill all with the same value => diff = 0
            return base_diff

        mn, mx = min(neighbors), max(neighbors)
        half_gap = (mx - mn + 1) // 2  # ceil((mx - mn)/2)

        return max(base_diff, half_gap)