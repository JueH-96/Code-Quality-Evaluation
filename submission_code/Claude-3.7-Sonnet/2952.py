class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        current_values = nums1.copy()
        current_sum = sum(current_values)
        
        # If the initial sum is already less than or equal to x, return 0
        if current_sum <= x:
            return 0
        
        # Simulate the process for each second
        for t in range(1, 10**6 + 1):  # Using a large upper bound
            # Increment all elements by their growth rates
            for i in range(n):
                current_values[i] += nums2[i]
                current_sum += nums2[i]
            
            # Find the element with the largest value to reset
            max_value = max(current_values)
            max_index = current_values.index(max_value)
            current_sum -= max_value
            current_values[max_index] = 0
            
            # Check if the sum is now less than or equal to x
            if current_sum <= x:
                return t
            
            # Optimization: check if sum can ever be reduced to x
            if t > 2*n:  # After enough iterations, check for divergence
                max_growth = max(nums2)
                total_growth = sum(nums2)
                if max_growth <= total_growth:  # If max element <= total growth, sum will never decrease
                    return -1
        
        return -1