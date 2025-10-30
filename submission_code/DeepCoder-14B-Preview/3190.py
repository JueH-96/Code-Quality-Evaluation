class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        min_swaps = float('inf')
        
        # Case 1: Do not swap the last elements
        a = nums1[-1]
        b = nums2[-1]
        swap_count = 0
        possible = True
        for i in range(n - 1):  # Process all elements except the last
            if nums1[i] <= a and nums2[i] <= b:
                continue
            else:
                if nums2[i] <= a and nums1[i] <= b:
                    swap_count += 1
                else:
                    possible = False
                    break
        if possible:
            min_swaps = min(min_swaps, swap_count)
        
        # Case 2: Swap the last elements
        a_case2 = nums2[-1]
        b_case2 = nums1[-1]
        swap_count_case2 = 1
        possible_case2 = True
        for i in range(n - 1):  # Process all elements except the last
            if nums1[i] <= a_case2 and nums2[i] <= b_case2:
                continue
            else:
                if nums2[i] <= a_case2 and nums1[i] <= b_case2:
                    swap_count_case2 += 1
                else:
                    possible_case2 = False
                    break
        if possible_case2:
            min_swaps = min(min_swaps, swap_count_case2)
        
        return min_swaps if min_swaps != float('inf') else -1