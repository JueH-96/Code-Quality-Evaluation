class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        import collections

        n = len(nums)
        
        # A map to store the indices of each unique number.
        # e.g., for nums = [1,2,1,2], positions will be {1: [0, 2], 2: [1, 3]}
        positions = collections.defaultdict(list)
        for i, num in enumerate(nums):
            positions[num].append(i)
            
        min_total_seconds = n # Initialize with a safe upper bound.

        # Calculate the time for each number to fill the array and find the minimum.
        for num_val in positions:
            indices = positions[num_val]
            k = len(indices)
            
            # The time is determined by the largest gap between consecutive occurrences.
            # Initialize with the wrap-around gap between the last and first occurrence.
            # The circular distance is (n - last_index) + first_index.
            max_gap = (n - indices[k-1]) + indices[0]
            
            # Iterate through the non-wrap-around gaps.
            for i in range(k - 1):
                gap = indices[i+1] - indices[i]
                if gap > max_gap:
                    max_gap = gap
            
            # Time to fill the largest gap of size `d` is floor(d/2),
            # as the number can propagate from both ends of the gap.
            seconds_for_this_num = max_gap // 2
            
            # Update the overall minimum.
            if seconds_for_this_num < min_total_seconds:
                min_total_seconds = seconds_for_this_num
                
        return min_total_seconds