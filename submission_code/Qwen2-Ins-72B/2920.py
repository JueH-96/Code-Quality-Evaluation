class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        d = defaultdict(list)
        n = len(nums)
        for i in range(n):
            d[nums[i]].append(i)
        ans = 0
        for k in d:
            m = len(d[k])
            if m == 1:
                ans = max(ans, (n-1)//2)
            else:
                d[k].append(d[k][0]+n)
                diff = [d[k][i+1]-d[k][i]-1 for i in range(m)]
                ans = max(ans, max(diff)//2)
        return ans