class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        vals = set(nums1 + nums2)
        n = len(nums1)
        min_ops = float('inf')
        for max1 in vals:
            for max2 in vals:
                possible = False
                swap_n1 = -1  # swap at position n -1: 0 or 1
                # Determine swap at position n -1
                if nums1[n -1] == max1 and nums2[n -1] == max2:
                    swap_n1 = 0
                elif nums1[n -1] == max2 and nums2[n -1] == max1:
                    swap_n1 = 1
                else:
                    continue  # Cannot get desired max1 and max2 at position n -1
                total_swaps = swap_n1  # 0 or 1
                possible = True
                for i in range(n -1):
                    # At index i, determine if we can satisfy the conditions
                    nums1_i_no_swap = nums1[i]
                    nums2_i_no_swap = nums2[i]
                    nums1_i_swap = nums2[i]
                    nums2_i_swap = nums1[i]
                    can_no_swap = (nums1_i_no_swap <= max1 and nums2_i_no_swap <= max2)
                    can_swap = (nums1_i_swap <= max1 and nums2_i_swap <= max2)
                    if can_no_swap and can_swap:
                        # We prefer not to swap to minimize swaps
                        continue  # swap[i]=0, cost 0
                    elif can_no_swap:
                        continue  # swap[i]=0, cost 0
                    elif can_swap:
                        total_swaps += 1  # swap[i]=1, cost 1
                    else:
                        possible = False
                        break  # Cannot satisfy conditions at index i
                if possible and total_swaps < min_ops:
                    min_ops = total_swaps
        return min_ops if min_ops != float('inf') else -1