class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Scenario 1: Keep last elements as they are
        max1_target = nums1[n-1]
        max2_target = nums2[n-1]
        swaps_scenario1 = 0
        possible_scenario1 = True
        
        for i in range(n-1):
            # Check if current configuration works
            if nums1[i] <= max1_target and nums2[i] <= max2_target:
                continue
            # Check if swapping would work
            elif nums2[i] <= max1_target and nums1[i] <= max2_target:
                swaps_scenario1 += 1
            else:
                possible_scenario1 = False
                break
        
        # Scenario 2: Swap last elements
        max1_target = nums2[n-1]
        max2_target = nums1[n-1]
        swaps_scenario2 = 1  # We already swapped the last elements
        possible_scenario2 = True
        
        for i in range(n-1):
            # Check if current configuration works (remember we swapped last)
            if nums1[i] <= max1_target and nums2[i] <= max2_target:
                continue
            # Check if swapping would work
            elif nums2[i] <= max1_target and nums1[i] <= max2_target:
                swaps_scenario2 += 1
            else:
                possible_scenario2 = False
                break
        
        # Return the minimum valid scenario
        if possible_scenario1 and possible_scenario2:
            return min(swaps_scenario1, swaps_scenario2)
        elif possible_scenario1:
            return swaps_scenario1
        elif possible_scenario2:
            return swaps_scenario2
        else:
            return -1