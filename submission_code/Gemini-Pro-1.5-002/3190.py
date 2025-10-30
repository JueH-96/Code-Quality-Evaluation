class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1 = 0
        max2 = 0
        for i in range(n):
            max1 = max(max1, nums1[i])
            max2 = max(max2, nums2[i])

        if nums1[n - 1] == max1 and nums2[n - 1] == max2:
            return 0

        ops = 0
        if nums1[n - 1] != max1 or nums2[n - 1] != max2:
            if nums1[n - 1] < max1 and nums2[n - 1] < max2:
                if nums2[n-1] >= max1 or nums1[n-1] >= max2 or max(nums1[n-1], nums2[n-1]) == max(max1, max2):
                    ops +=1
                    nums1[n-1], nums2[n-1] = nums2[n-1], nums1[n-1]
                else:
                    return -1
            elif nums1[n - 1] < max1:
                if nums2[n - 1] >= max1:
                    ops += 1
                    nums1[n - 1], nums2[n - 1] = nums2[n - 1], nums1[n - 1]
                else:
                    
                    found = False
                    for i in range(n-1):
                        if nums1[i] >= max1 and nums2[i] <= nums2[n-1]:
                            ops += 1
                            nums1[i], nums1[n-1] = nums1[n-1], nums1[i]
                            found = True
                            break
                        elif nums2[i] >= max1:
                            ops += 1
                            nums1[n-1], nums2[i] = nums2[i], nums1[n-1]
                            found = True
                            break
                    if not found:
                        return -1
            elif nums2[n - 1] < max2:
                if nums1[n - 1] >= max2:
                    ops += 1
                    nums1[n - 1], nums2[n - 1] = nums2[n - 1], nums1[n - 1]
                else:
                    found = False
                    for i in range(n-1):
                        if nums2[i] >= max2 and nums1[i] <= nums1[n-1]:
                            ops += 1
                            nums2[i], nums2[n-1] = nums2[n-1], nums2[i]
                            found = True
                            break
                        elif nums1[i] >= max2:
                            ops += 1
                            nums2[n-1], nums1[i] = nums1[i], nums2[n-1]
                            found = True
                            break
                    if not found:
                        return -1
        
        max1 = 0
        max2 = 0
        for i in range(n):
            max1 = max(max1, nums1[i])
            max2 = max(max2, nums2[i])
        
        if nums1[n-1] != max1 or nums2[n-1] != max2:
            return -1
        
        return ops