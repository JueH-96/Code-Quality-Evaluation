class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = Counter(nums)
        ans = 0
        for x in sorted(count.keys()):
            if count[x] == 0:
                continue
            ans += count[x]
            count[x] = 0
            curr = x * x
            while curr in count and count[curr] > 0:
                ans += min(count[x], count[curr])
                count[curr] -= min(count[x], count[curr])
                x = curr
                curr = x * x

        return ans