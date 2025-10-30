from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        pos = defaultdict(list)
        for i, num in enumerate(nums):
            pos[num].append(i)
        
        min_time = float('inf')
        
        for key in pos:
            positions = sorted(pos[key])
            m = len(positions)
            
            if m == 1:
                gap = n - 1
                current_max = (gap + 1) // 2
            else:
                current_max = 0
                # Calculate gaps between consecutive positions
                for i in range(1, m):
                    gap = positions[i] - positions[i-1] - 1
                    steps = (gap + 1) // 2
                    if steps > current_max:
                        current_max = steps
                # Calculate the circular gap
                gap = (positions[0] + n - positions[-1] - 1)
                steps = (gap + 1) // 2
                if steps > current_max:
                    current_max = steps
            
            if current_max < min_time:
                min_time = current_max
        
        return min_time if min_time != float('inf') else 0