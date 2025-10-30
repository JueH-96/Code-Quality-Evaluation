class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for num in nums:
            found = False
            for x in range(1, num + 1):
                if (x | (x + 1)) == num:
                    ans.append(x)
                    found = True
                    break
            if not found:
                ans.append(-1)
        return ans