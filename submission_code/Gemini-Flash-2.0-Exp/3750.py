class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        ans = []
        for q in queries:
            target = nums[q]
            min_dist = float('inf')
            found = False
            for i in range(n):
                if i != q and nums[i] == target:
                    dist1 = abs(i - q)
                    dist2 = n - dist1
                    min_dist = min(min_dist, min(dist1, dist2))
                    found = True
            if found:
                ans.append(min_dist)
            else:
                ans.append(-1)
        return ans