from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # Helper function to compute minimum swap cost for indices [0, n-2]
        # given that the fixed last index's chosen final values are (f1, f2)
        # For each index i, we check:
        #   Option 0 (no swap): use (nums1[i], nums2[i]) must satisfy nums1[i] <= f1 and nums2[i] <= f2.
        #   Option 1 (swap): use (nums2[i], nums1[i]) must satisfy nums2[i] <= f1 and nums1[i] <= f2.
        # Then we add the minimal cost from these options.
        def compute_cost(f1_last: int, f2_last: int) -> int:
            total = 0
            for i in range(n - 1):
                cost_options = []
                # Option: not swap (cost = 0)
                if nums1[i] <= f1_last and nums2[i] <= f2_last:
                    cost_options.append(0)
                # Option: swap (cost = 1)
                if nums2[i] <= f1_last and nums1[i] <= f2_last:
                    cost_options.append(1)
                # If neither option works, it's impossible under this branch.
                if not cost_options:
                    return float('inf')
                total += min(cost_options)
            return total

        # Branch 1: Do NOT swap at index n-1.
        # Final values at index n-1 become:
        #   nums1[n-1] = nums1[n-1], nums2[n-1] = nums2[n-1]
        branch1_cost = compute_cost(nums1[n-1], nums2[n-1])
        # No extra cost added for index n-1 (stays same).

        # Branch 2: SWAP at index n-1.
        # Final values at index n-1 become:
        #   nums1[n-1] = nums2[n-1], nums2[n-1] = nums1[n-1]
        branch2_cost = compute_cost(nums2[n-1], nums1[n-1])
        if branch2_cost != float('inf'):
            branch2_cost += 1  # cost for swapping index n-1

        # Choose the minimum cost among branches.
        ans = min(branch1_cost, branch2_cost)
        return ans if ans != float('inf') else -1