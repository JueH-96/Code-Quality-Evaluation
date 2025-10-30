class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        cnt = [0]
        ans = 0
        invalid = 0
        def check(x):
            nonlocal invalid
            if x == 0:
                invalid = 0
                return k * (cnt[x] - 1) + nums[cnt[x] - 1] - nums[0] <= 2 * k
            invalid = cnt[x - 1]
            if k * (cnt[x] - invalid - 1) + nums[cnt[x] - 1] - nums[invalid] > 2 * k:
                return False
            while invalid > 0 and nums[invalid - 1] - nums[cnt[x] - invalid - 1] <= 2 * k:
                invalid -= 1
            return True
        for i, x in enumerate(nums):
            while cnt[-1] < i + 1 and (cnt[-1] == 0 or x - nums[cnt[-1] - 1] > 2 * k):
                bisect.insort_right(cnt, bisect.bisect_left(cnt, i, key=check) - 1)
            invalid += i - cnt[-1]
            ans = max(ans, cnt[-1] - invalid + 1)
        return ans