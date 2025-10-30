class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        diff = [target[i] - nums[i] for i in range(len(nums))]
        ans = 0
        for i in range(len(diff)):
            if i == 0:
                ans += abs(diff[i])
            else:
                if diff[i] * diff[i - 1] > 0:
                    if abs(diff[i]) > abs(diff[i - 1]):
                        ans += abs(diff[i] - diff[i - 1])
                else:
                    ans += abs(diff[i])
        return ans