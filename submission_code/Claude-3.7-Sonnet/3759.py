from heapq import nlargest

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = [0] * n
        
        for i in range(n):
            # Find all values from nums2 at indices j where nums1[j] < nums1[i]
            values = [nums2[j] for j in range(n) if nums1[j] < nums1[i]]
            
            # If we found any valid values, take the sum of the k largest
            if values:
                answer[i] = sum(nlargest(min(k, len(values)), values))
                
        return answer