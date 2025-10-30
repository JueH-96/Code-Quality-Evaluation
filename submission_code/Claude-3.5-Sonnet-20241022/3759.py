class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = []
        
        # For each index i
        for i in range(n):
            # Find all indices where nums1[j] < nums1[i]
            smaller_indices = []
            for j in range(n):
                if nums1[j] < nums1[i]:
                    smaller_indices.append(j)
            
            # If no indices found, append 0
            if not smaller_indices:
                answer.append(0)
                continue
            
            # Get nums2 values at these indices
            values = [nums2[j] for j in smaller_indices]
            # Sort in descending order
            values.sort(reverse=True)
            
            # Take sum of k largest values (or all if less than k available)
            k_sum = sum(values[:min(k, len(values))])
            answer.append(k_sum)
        
        return answer