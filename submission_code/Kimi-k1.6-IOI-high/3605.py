class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for y in nums:
            found = False
            for x in range(y):
                if (x | (x + 1)) == y:
                    ans.append(x)
                    found = True
                    break
            if not found:
                ans.append(-1)
        return ans