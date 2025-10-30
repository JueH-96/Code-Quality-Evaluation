class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = []
        
        for i in range(n):
            # Find all nums2[j] where nums1[j] < nums1[i]
            valid_nums2 = []
            for j in range(n):
                if nums1[j] < nums1[i]:
                    valid_nums2.append(nums2[j])
            
            # Sort in descending order and take at most k values
            valid_nums2.sort(reverse=True)
            
            # Sum the top k values (or all if less than k available)
            if valid_nums2:
                sum_val = sum(valid_nums2[:min(k, len(valid_nums2))])
            else:
                sum_val = 0
                
            answer.append(sum_val)
        
        return answer