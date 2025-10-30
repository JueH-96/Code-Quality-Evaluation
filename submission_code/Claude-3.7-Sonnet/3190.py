class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Scenario A: Keep the last elements in their original positions
        max1_a = nums1[n-1]
        max2_a = nums2[n-1]
        ops_a = 0
        
        for i in range(n-1):
            need_swap1 = nums1[i] > max1_a
            need_swap2 = nums2[i] > max2_a
            
            # If swapping won't help for either condition, scenario A is impossible
            if (need_swap1 and nums2[i] > max1_a) or (need_swap2 and nums1[i] > max2_a):
                ops_a = -1
                break
            elif need_swap1 or need_swap2:
                ops_a += 1
        
        # Scenario B: Swap the last elements between arrays
        max1_b = nums2[n-1]
        max2_b = nums1[n-1]
        ops_b = 1  # Start with 1 for swapping the last elements
        
        for i in range(n-1):
            need_swap1 = nums1[i] > max1_b
            need_swap2 = nums2[i] > max2_b
            
            # If swapping won't help for either condition, scenario B is impossible
            if (need_swap1 and nums2[i] > max1_b) or (need_swap2 and nums1[i] > max2_b):
                ops_b = -1
                break
            elif need_swap1 or need_swap2:
                ops_b += 1
        
        # Return the minimum operations required or -1 if both scenarios are impossible
        if ops_a == -1 and ops_b == -1:
            return -1
        elif ops_a == -1:
            return ops_b
        elif ops_b == -1:
            return ops_a
        else:
            return min(ops_a, ops_b)