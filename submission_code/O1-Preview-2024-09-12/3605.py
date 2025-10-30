class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        nums_to_min_ans = {}
        for A in range(0, 1001):
            result = A | (A+1)
            if result > 1000:
                continue
            if result not in nums_to_min_ans or A < nums_to_min_ans[result]:
                nums_to_min_ans[result] = A

        ans = []
        for num in nums:
            if num in nums_to_min_ans:
                ans.append(nums_to_min_ans[num])
            else:
                ans.append(-1)
        return ans