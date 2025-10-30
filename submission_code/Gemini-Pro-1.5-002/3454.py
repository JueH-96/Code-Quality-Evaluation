class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        diff = [target[i] - nums[i] for i in range(n)]
        ans = 0
        curr = 0
        for i in range(n):
            if diff[i] == 0:
                continue
            if curr == 0:
                curr = diff[i]
                ans += abs(curr)
            elif (diff[i] > 0 and curr > 0) or (diff[i] < 0 and curr < 0):
                curr = diff[i]
                ans += abs(curr - diff[i])

            else:
                ans += abs(curr)
                curr = diff[i]
                ans += abs(curr)


        return ans