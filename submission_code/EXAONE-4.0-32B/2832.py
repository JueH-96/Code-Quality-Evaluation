from collections import defaultdict
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        groups = defaultdict(list)
        for i, num in enumerate(nums):
            groups[num].append(i)
        
        ans = 0
        for x, positions in groups.items():
            n_pos = len(positions)
            if n_pos == 0:
                continue
            Q = [positions[i] - i for i in range(n_pos)]
            j = 0
            for i in range(n_pos):
                while j < n_pos and Q[j] <= Q[i] + k:
                    j += 1
                current_length = j - i
                if current_length > ans:
                    ans = current_length
        return ans