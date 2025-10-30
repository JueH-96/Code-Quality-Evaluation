class Solution:
    def maxSum(self, nums: List[int]) -> int:
        d = defaultdict(list)
        for n in nums:
            d[max(str(n))].append(n)
        ans = -1
        for v in d.values():
            if len(v) > 1:
                ans = max(ans, sum(sorted(v)[-2:]))
        return ans