class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Possibility 1: Don't swap at position n-1
        target1_1 = nums1[n-1]
        target2_1 = nums2[n-1]
        ops1 = 0
        possible1 = True
        
        for i in range(n-1):
            if nums1[i] <= target1_1 and nums2[i] <= target2_1:
                # No swap needed
                pass
            elif nums2[i] <= target1_1 and nums1[i] <= target2_1:
                # Must swap
                ops1 += 1
            else:
                # Impossible
                possible1 = False
                break
        
        # Possibility 2: Swap at position n-1
        target1_2 = nums2[n-1]
        target2_2 = nums1[n-1]
        ops2 = 1  # We swap at position n-1
        possible2 = True
        
        for i in range(n-1):
            if nums1[i] <= target1_2 and nums2[i] <= target2_2:
                # No swap needed
                pass
            elif nums2[i] <= target1_2 and nums1[i] <= target2_2:
                # Must swap
                ops2 += 1
            else:
                # Impossible
                possible2 = False
                break
        
        if possible1 and possible2:
            return min(ops1, ops2)
        elif possible1:
            return ops1
        elif possible2:
            return ops2
        else:
            return -1