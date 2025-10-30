class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        ans = float('inf')
        for i in range(1 << n):
            temp_nums1 = nums1[:]
            temp_nums2 = nums2[:]
            count = 0
            for j in range(n):
                if (i >> j) & 1:
                    temp_nums1[j], temp_nums2[j] = temp_nums2[j], temp_nums1[j]
                    count += 1
            
            max_nums1 = max(temp_nums1)
            max_nums2 = max(temp_nums2)
            
            if temp_nums1[-1] == max_nums1 and temp_nums2[-1] == max_nums2:
                ans = min(ans, count)
        
        if ans == float('inf'):
            return -1
        else:
            return ans