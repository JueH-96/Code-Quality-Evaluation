class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Step 1: Find first and last occurrence of each unique number
        first = {}
        last = {}
        for i, num in enumerate(nums):
            if num not in first:
                first[num] = i
            last[num] = i
        
        # Step 2: Create intervals for each unique number
        intervals = []
        for num in first:
            intervals.append((first[num], last[num]))
        
        # Step 3: Sort the intervals by their start index
        intervals.sort()
        
        # Step 4: Merge overlapping or adjacent intervals
        if not intervals:
            return 1  # Only one way to partition an empty array
        
        merged = []
        current_start, current_end = intervals[0]
        for i in range(1, len(intervals)):
            start, end = intervals[i]
            if start < current_end:
                # Merge intervals
                current_end = max(current_end, end)
            else:
                # Add the current merged interval to the list
                merged.append((current_start, current_end))
                # Start a new interval
                current_start, current_end = start, end
        # Add the last merged interval
        merged.append((current_start, current_end))
        
        # Step 5: Count the number of merged intervals, k
        k = len(merged)
        
        # Step 6: Compute 2^(k-1) modulo 10^9+7
        if k == 0:
            return 1  # Only one way to partition an empty array
        answer = pow(2, k - 1, MOD)
        return answer