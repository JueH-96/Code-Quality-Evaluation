class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        
        # Initialize for i=0
        prev_a_last = nums1[0]
        prev_a_length = 1
        prev_b_last = nums2[0]
        prev_b_length = 1
        
        max_length = 1
        
        for i in range(1, n):
            a = nums1[i]
            b = nums2[i]
            
            current_a_last = a
            current_a_length = 1
            
            # Check if current a can extend from a's previous
            if a >= prev_a_last:
                current_a_length = max(current_a_length, prev_a_length + 1)
            
            # Check if current a can extend from b's previous
            if a >= prev_b_last:
                current_a_length = max(current_a_length, prev_b_length + 1)
            
            current_b_last = b
            current_b_length = 1
            
            # Check if current b can extend from a's previous
            if b >= prev_a_last:
                current_b_length = max(current_b_length, prev_a_length + 1)
            
            # Check if current b can extend from b's previous
            if b >= prev_b_last:
                current_b_length = max(current_b_length, prev_b_length + 1)
            
            # Update max_length
            current_max = max(current_a_length, current_b_length)
            if current_max > max_length:
                max_length = current_max
            
            # Update previous values for next iteration
            prev_a_last = current_a_last
            prev_a_length = current_a_length
            prev_b_last = current_b_last
            prev_b_length = current_b_length
        
        return max_length