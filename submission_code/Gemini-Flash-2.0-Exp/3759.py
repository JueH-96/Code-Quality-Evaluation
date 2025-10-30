class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = []
        for i in range(n):
            indices = []
            for j in range(n):
                if nums1[j] < nums1[i]:
                    indices.append(j)
            
            values = []
            for index in indices:
                values.append(nums2[index])
            
            values.sort(reverse=True)
            
            total_sum = 0
            for l in range(min(k, len(values))):
                total_sum += values[l]
            
            answer.append(total_sum)
        
        return answer