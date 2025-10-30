class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0  # As per problem constraints, n is at least 1
        
        # Scenario 1: Do not swap the last elements
        a1, b1 = nums1[-1], nums2[-1]
        count1 = 0
        valid1 = True
        for i in range(n - 1):
            x, y = nums1[i], nums2[i]
            if x <= a1 and y <= b1:
                continue
            elif y <= a1 and x <= b1:
                count1 += 1
            else:
                valid1 = False
                break
        
        # Scenario 2: Swap the last elements
        a2, b2 = nums2[-1], nums1[-1]
        count2 = 1  # Account for swapping the last element
        valid2 = True
        for i in range(n - 1):
            x, y = nums1[i], nums2[i]
            if x <= a2 and y <= b2:
                continue
            elif y <= a2 and x <= b2:
                count2 += 1
            else:
                valid2 = False
                break
        
        # Collect results from valid scenarios
        res = []
        if valid1:
            res.append(count1)
        if valid2:
            res.append(count2)
        
        return min(res) if res else -1