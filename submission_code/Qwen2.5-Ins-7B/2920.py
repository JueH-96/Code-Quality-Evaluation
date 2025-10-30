from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        index_map = defaultdict(list)
        
        for i, num in enumerate(nums):
            index_map[num].append(i)
        
        min_seconds = float('inf')
        
        for indices in index_map.values():
            max_distance = 0
            for i in range(len(indices) - 1):
                max_distance = max(max_distance, indices[i + 1] - indices[i] - 1)
            max_distance = max(max_distance, n - 1 + indices[0] - indices[-1])
            min_seconds = min(min_seconds, (max_distance + 1) // 2)
        
        return min_seconds