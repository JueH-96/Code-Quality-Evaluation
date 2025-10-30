class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        # If all numbers are already equal, no operations are needed
        if all(x == nums[0] for x in nums):
            return 0
        
        from collections import defaultdict
        
        n = len(nums)
        positions_map = defaultdict(list)
        
        # Build a map from value -> list of indices where that value appears
        for i, val in enumerate(nums):
            positions_map[val].append(i)
        
        answer = n  # Initialize answer to a large number
        
        # For each distinct value, compute the minimum time needed to make the whole array that value
        for val, positions in positions_map.items():
            positions.sort()
            
            # Find the largest gap between consecutive occurrences (in a circular sense)
            max_gap = 0
            for i in range(len(positions) - 1):
                max_gap = max(max_gap, positions[i+1] - positions[i])
            
            # Include the wrap-around gap from last to first + n
            max_gap = max(max_gap, positions[0] + n - positions[-1])
            
            # The time needed is half of the largest gap (integer division)
            time_needed = max_gap // 2
            
            answer = min(answer, time_needed)
        
        return answer