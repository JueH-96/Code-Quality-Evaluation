class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        groups = {}
        for i in range(n):
            num = nums[i]
            group_key = 1
            d = 2
            while d * d <= num:
                cnt = 0
                while num % d == 0:
                    num //= d
                    cnt += 1
                if cnt % 2 == 1:
                    group_key *= d
                d += 1
            if num > 1:
                group_key *= num
            if group_key not in groups:
                groups[group_key] = 0
            groups[group_key] += nums[i]
        
        ans = 0
        for val in groups.values():
            ans = max(ans, val)
        return ans