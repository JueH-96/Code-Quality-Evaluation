import math

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        cost1 = 0
        possible1 = True
        target_u1_1 = nums1[n-1]
        target_v1_1 = nums2[n-1]
        for i in range(n - 1):
            u_i = nums1[i]
            v_i = nums2[i]
            if u_i <= target_u1_1 and v_i <= target_v1_1:
                continue
            elif v_i <= target_u1_1 and u_i <= target_v1_1:
                cost1 += 1
            else:
                possible1 = False
                break
        if not possible1:
            cost1 = float('inf')
            
        cost2 = 1
        possible2 = True
        target_u1_2 = nums2[n-1]
        target_v1_2 = nums1[n-1]
        for i in range(n - 1):
            u_i = nums1[i]
            v_i = nums2[i]
            if u_i <= target_u1_2 and v_i <= target_v1_2:
                continue
            elif v_i <= target_u1_2 and u_i <= target_v1_2:
                cost2 += 1
            else:
                possible2 = False
                break
        if not possible2:
            cost2 = float('inf')
            
        min_cost = min(cost1, cost2)
        if min_cost == float('inf'):
            return -1
        else:
            return min_cost