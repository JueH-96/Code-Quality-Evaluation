class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_sum = sum(nums1)
        
        # If the initial sum is already less than or equal to x, return 0
        if total_sum <= x:
            return 0
        
        # Calculate the maximum possible increments after t seconds
        # and the minimum time required to achieve the goal
        time = 0
        while True:
            time += 1
            # Increment nums1 by nums2
            for i in range(n):
                nums1[i] += nums2[i]
            
            # After incrementing, we can choose to set one element to 0
            # Calculate the new total sum
            total_sum = sum(nums1)
            
            # Check if we can achieve the goal by setting one element to 0
            if total_sum <= x:
                return time
            
            # Check if we can reduce the total sum by setting the maximum element to 0
            max_increment = max(nums1)
            total_sum -= max_increment
            
            # If even after setting the maximum to 0, we can't achieve the goal
            if total_sum > x:
                # Check if we can ever reach the goal
                if all(num + time * inc > x for num, inc in zip(nums1, nums2)):
                    return -1
            
            # If we reach here, we continue to the next second