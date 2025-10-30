class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        d = {}
        for i in range(n):
            if nums[i] in d:
                d[nums[i]].append(i)
            else:
                d[nums[i]] = [i]
        ans = 1
        for i in range(n):
            x = nums[i]
            k = 1
            cur = 1
            while x**k in d:
                if len(d[x**k]) > 0 and d[x**k][-1] > i:
                    cur += 2
                    k *= 2
                else:
                    break
            if x**k in d and len(d[x**k]) > 0 and d[x**k][0] > i:
                cur -= 1
            ans = max(ans, cur)
        return ans