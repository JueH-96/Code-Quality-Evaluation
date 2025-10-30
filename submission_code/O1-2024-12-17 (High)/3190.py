class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Helper to compute cost if we want final F1[n-1] = lastF1, F2[n-1] = lastF2,
        # and we add swapCostForLast to the final answer if we must swap index n-1.
        def compute_cost(A1, A2, lastF1, lastF2, swapCostForLast):
            cost = 0
            # Check indices 0..n-2
            for i in range(n - 1):
                not_swap_ok = (A1[i] <= lastF1) and (A2[i] <= lastF2)
                swap_ok = (A2[i] <= lastF1) and (A1[i] <= lastF2)
                
                # If neither configuration works, impossible
                if not not_swap_ok and not swap_ok:
                    return None
                
                # If only not-swap works, cost += 0
                if not_swap_ok and not swap_ok:
                    continue
                
                # If only swap works, cost += 1
                if swap_ok and not not_swap_ok:
                    cost += 1
                    continue
                
                # If both work, pick the cheaper (not swapping costs 0)
                # so we do nothing extra here.
            
            return cost + swapCostForLast
        
        # Case A: Do not swap at index n-1
        # => F1[n-1] = nums1[n-1], F2[n-1] = nums2[n-1]
        costA = compute_cost(nums1, nums2, nums1[n-1], nums2[n-1], 0)
        
        # Case B: Swap at index n-1
        # => F1[n-1] = nums2[n-1], F2[n-1] = nums1[n-1]
        costB = compute_cost(nums1, nums2, nums2[n-1], nums1[n-1], 1)
        
        # Determine the answer
        if costA is None and costB is None:
            return -1
        if costA is not None and costB is not None:
            return min(costA, costB)
        return costA if costB is None else costB