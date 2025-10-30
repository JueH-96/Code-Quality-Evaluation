class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        import heapq
        
        n = len(nums1)
        # Calculate the initial sum
        current_sum = sum(nums1)
        
        # If the initial sum is already less than or equal to x, return 0
        if current_sum <= x:
            return 0
        
        # Priority queue to store the most effective reductions
        # We use negative because heapq is a min-heap by default, we need max-heap behavior
        max_heap = []
        
        # Calculate the potential impact of setting each nums1[i] to 0 after each second
        for i in range(n):
            if nums2[i] == 0:
                # If nums2[i] is 0, nums1[i] will not increase, so the effect is constant
                heapq.heappush(max_heap, (-nums1[i], i))
            else:
                # Calculate the total reduction possible by setting nums1[i] to 0 at each second
                # The reduction is the current value plus the growth that would have happened
                # We need to consider the effect over time, so we simulate it second by second
                initial_value = nums1[i]
                growth_rate = nums2[i]
                time = 0
                while initial_value + growth_rate * time > 0:
                    effect = initial_value + growth_rate * time
                    heapq.heappush(max_heap, (-effect, i, time))
                    time += 1
        
        # Try to reduce the sum below or equal to x using the least time possible
        time_elapsed = 0
        while max_heap and current_sum > x:
            effect, idx, time_to_apply = heapq.heappop(max_heap)
            effect = -effect
            
            # Calculate the actual value at the time we apply the zeroing
            actual_value = nums1[idx] + nums2[idx] * time_to_apply
            if actual_value > 0:
                current_sum -= actual_value
                nums1[idx] = 0  # After setting to zero, it stops contributing to the sum
                time_elapsed = max(time_elapsed, time_to_apply + 1)
        
        # If we managed to reduce the sum to a value <= x, return the time elapsed
        if current_sum <= x:
            return time_elapsed
        
        # If not possible, return -1
        return -1