class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        
        # Sort elements by nums2[i] in descending order
        order = sorted(range(n), key=lambda i: nums2[i], reverse=True)
        
        # Try different total times T
        for T in range(n + 1):
            # Calculate the sum without any operations
            total_sum = sum(nums1[i] + T * nums2[i] for i in range(n))
            
            if total_sum <= x:
                return T
            
            # Calculate the maximum benefit by setting min(T, n) elements to 0
            max_benefit = 0
            for j in range(min(T, n)):
                i = order[j]
                t = T - j  # time slot T, T-1, ..., T-min(T,n)+1
                max_benefit += nums1[i] + t * nums2[i]
            
            if total_sum - max_benefit <= x:
                return T
        
        return -1