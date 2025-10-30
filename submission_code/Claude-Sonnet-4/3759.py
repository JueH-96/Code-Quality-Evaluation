class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = []
        
        for i in range(n):
            # Find all indices j where nums1[j] < nums1[i]
            candidates = []
            for j in range(n):
                if nums1[j] < nums1[i]:
                    candidates.append(nums2[j])
            
            # Sort candidates in descending order to get largest values first
            candidates.sort(reverse=True)
            
            # Take at most k values and sum them
            max_sum = sum(candidates[:k])
            answer.append(max_sum)
        
        return answer