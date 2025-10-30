class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        num_positions = {}
        
        # Store positions of each number
        for i, num in enumerate(nums):
            if num not in num_positions:
                num_positions[num] = []
            num_positions[num].append(i)
        
        min_seconds = float('inf')
        
        for num, positions in num_positions.items():
            positions.append(positions[0] + n)  # Add first position to end for circular array
            max_gap = 0
            
            for i in range(1, len(positions)):
                gap = positions[i] - positions[i-1]
                max_gap = max(max_gap, gap)
            
            seconds = (max_gap + 1) // 2
            min_seconds = min(min_seconds, seconds)
        
        return min_seconds