class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        import math
        
        n = len(nums1)
        
        # We'll try two scenarios for (nums1[n - 1], nums2[n - 1]):
        # 1) No swap at index n-1
        # 2) Swap at index n-1
        #
        # Then, for i in [0..n-2], we decide whether to swap or not swap
        # in a way that ensures:
        #   final_nums1[i] <= final_nums1[n-1] and final_nums2[i] <= final_nums2[n-1]
        # with the minimum total swaps.
        
        def check_feasibility(c1, c2):
            """
            Checks if it's possible for final nums1[i] <= c1 and final nums2[i] <= c2
            for all i in [0..n-2], and returns the minimal number of swaps needed.
            If impossible, returns math.inf.
            """
            swaps = 0
            for i in range(n - 1):
                can_no_swap = (nums1[i] <= c1 and nums2[i] <= c2)
                can_swap    = (nums2[i] <= c1 and nums1[i] <= c2)
                
                if not can_no_swap and not can_swap:
                    # No way to keep both numbers <= (c1, c2) whether we swap or not
                    return math.inf
                if can_no_swap and not can_swap:
                    # Must NOT swap
                    continue
                if not can_no_swap and can_swap:
                    # Must swap
                    swaps += 1
                # if can_no_swap and can_swap:
                # We prefer no swap (cost = 0) over swap (cost = 1),
                # so do nothing (no increment).
            return swaps
        
        # Scenario 1: no swap at index n - 1
        c1a, c2a = nums1[n-1], nums2[n-1]
        costA = check_feasibility(c1a, c2a)
        
        # Scenario 2: swap at index n - 1
        c1b, c2b = nums2[n-1], nums1[n-1]
        costB = check_feasibility(c1b, c2b)
        if costB != math.inf:
            costB += 1  # Must swap at index n-1
        
        ans = min(costA, costB)
        return ans if ans != math.inf else -1