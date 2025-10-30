class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = [0] * n
        
        for i in range(n):
            indices = []
            for j in range(n):
                if nums1[j] < nums1[i]:
                    indices.append(j)
            
            indices.sort(key=lambda x: nums2[x], reverse=True)
            total = 0
            for j in range(min(k, len(indices))):
                total += nums2[indices[j]]
            answer[i] = total
        
        return answer