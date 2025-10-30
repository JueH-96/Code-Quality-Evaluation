class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        import heapq
        
        n = len(nums1)
        answer = []
        
        for i in range(n):
            # Find all nums2[j] where nums1[j] < nums1[i]
            candidates = []
            for j in range(n):
                if nums1[j] < nums1[i]:
                    candidates.append(nums2[j])
            
            # If no candidates, sum is 0
            if not candidates:
                answer.append(0)
            else:
                # Use heapq.nlargest to efficiently find k largest values
                largest_k = heapq.nlargest(k, candidates)
                answer.append(sum(largest_k))
        
        return answer