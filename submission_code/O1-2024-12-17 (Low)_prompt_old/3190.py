class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Helper function:
        # Given desired "cap1" = final1[n-1], "cap2" = final2[n-1],
        # returns the minimum number of swaps needed (over indices 0..n-2)
        # so that for every i in [0..n-2],
        # final1[i] <= cap1 and final2[i] <= cap2.
        # "final1[i]" is nums1[i] if we do NOT swap index i, or nums2[i] if we DO swap index i.
        # "final2[i]" is nums2[i] if we do NOT swap index i, or nums1[i] if we DO swap index i.
        # If it's impossible, return -1.
        def check(cap1, cap2):
            swaps = 0
            for i in range(n - 1):
                no_swap_ok = (nums1[i] <= cap1 and nums2[i] <= cap2)
                swap_ok    = (nums2[i] <= cap1 and nums1[i] <= cap2)
                if not no_swap_ok and not swap_ok:
                    return -1  # impossible for index i
                # If both options are valid, prefer no swap (swaps += 0)
                # If only swap is valid, we must swap (swaps += 1)
                if not no_swap_ok and swap_ok:
                    swaps += 1
                # if both are valid or only no_swap is valid, do nothing extra
            return swaps
        
        # Case A: do NOT swap the last index
        # final1[n-1] = nums1[n-1], final2[n-1] = nums2[n-1]
        capA1 = nums1[-1]
        capA2 = nums2[-1]
        costA = check(capA1, capA2)
        
        # Case B: DO swap the last index
        # final1[n-1] = nums2[n-1], final2[n-1] = nums1[n-1]
        # This may or may not cost 1 operation depending on whether the values are already equal.
        capB1 = nums2[-1]
        capB2 = nums1[-1]
        costB_check = check(capB1, capB2)
        
        # If it's possible, we add the cost of swapping the last index
        # (which is 1 if the last elements differ, 0 if they're the same).
        if costB_check != -1:
            if nums1[-1] == nums2[-1]:
                costB = costB_check  # swapping the same value doesn't really change anything
            else:
                costB = costB_check + 1
        else:
            costB = -1
        
        # Combine results:
        if costA == -1 and costB == -1:
            return -1
        elif costA == -1:
            return costB
        elif costB == -1:
            return costA
        else:
            return min(costA, costB)