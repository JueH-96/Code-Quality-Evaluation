class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
            
        # Create map of value to indices where it appears
        pos = {}
        for i, num in enumerate(nums):
            if num not in pos:
                pos[num] = []
            pos[num].append(i)
            
        min_seconds = float('inf')
        
        # For each unique value
        for num in pos:
            indices = pos[num]
            indices.append(indices[0] + n)  # Add first index + n for circular array
            
            # Find max gap between consecutive occurrences
            max_gap = 0
            for i in range(1, len(indices)):
                gap = indices[i] - indices[i-1]
                max_gap = max(max_gap, gap)
                
            # We need ceil(max_gap/2) seconds to spread this number
            seconds = (max_gap + 1) // 2
            min_seconds = min(min_seconds, seconds)
            
        return min_seconds