class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        def solve(target_max1, target_max2):
            cost = 0
            for i in range(n-1):
                # Option 1: Don't swap
                option1_valid = nums1[i] <= target_max1 and nums2[i] <= target_max2
                # Option 2: Swap
                option2_valid = nums2[i] <= target_max1 and nums1[i] <= target_max2
                
                if option1_valid:
                    # We can choose not to swap (cost 0)
                    pass
                elif option2_valid:
                    # We must swap (cost 1)
                    cost += 1
                else:
                    # Neither option works
                    return -1
            return cost
        
        # Case 1: nums1[n-1] stays in nums1, nums2[n-1] stays in nums2
        cost1 = solve(nums1[n-1], nums2[n-1])
        
        # Case 2: nums1[n-1] goes to nums2, nums2[n-1] goes to nums1
        cost2 = solve(nums2[n-1], nums1[n-1])
        if cost2 != -1:
            cost2 += 1  # Add the cost of swapping at position n-1
        
        if cost1 == -1 and cost2 == -1:
            return -1
        elif cost1 == -1:
            return cost2
        elif cost2 == -1:
            return cost1
        else:
            return min(cost1, cost2)