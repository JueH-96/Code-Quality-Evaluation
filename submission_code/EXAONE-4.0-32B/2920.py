from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        d = defaultdict(list)
        for i, num in enumerate(nums):
            d[num].append(i)
        
        if n == 0:
            return 0
        
        ans = float('inf')
        for key, indices in d.items():
            k = len(indices)
            if k == n:
                return 0
            gaps = []
            for i in range(k):
                j = (i + 1) % k
                gap = indices[j] - indices[i] - 1
                if gap < 0:
                    gap += n
                gaps.append(gap)
            max_time = max((gap + 1) // 2 for gap in gaps)
            if max_time < ans:
                ans = max_time
        return ans