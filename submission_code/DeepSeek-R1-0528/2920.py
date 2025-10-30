from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        indices_map = defaultdict(list)
        for i, num in enumerate(nums):
            indices_map[num].append(i)
        
        ans = float('inf')
        for num, indices in indices_map.items():
            m = len(indices)
            dup = indices + [i + n for i in indices]
            max_time = 0
            for j in range(m):
                gap = dup[j + 1] - dup[j]
                time = gap // 2
                if time > max_time:
                    max_time = time
            ans = min(ans, max_time)
        return ans