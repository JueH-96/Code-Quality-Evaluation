from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        groups = defaultdict(list)
        for i, num in enumerate(nums):
            groups[num].append(i)
        
        ans = 0
        for pos in groups.values():
            l = 0
            n_pos = len(pos)
            for r in range(n_pos):
                while (pos[r] - pos[l] - (r - l)) > k:
                    l += 1
                ans = max(ans, r - l + 1)
        return ans