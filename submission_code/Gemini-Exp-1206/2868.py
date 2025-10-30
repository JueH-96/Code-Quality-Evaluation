class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                is_continuous = True
                for i1 in range(i, j + 1):
                    for i2 in range(i, j + 1):
                        if abs(nums[i1] - nums[i2]) > 2:
                            is_continuous = False
                            break
                    if not is_continuous:
                        break
                if is_continuous:
                    ans += 1
        return ans