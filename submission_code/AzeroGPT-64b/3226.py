class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        ans = []
        while nums:
            ans.append(nums.pop(1))
            ans.append(nums.pop(0))
        return ans