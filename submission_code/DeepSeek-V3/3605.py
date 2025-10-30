class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for num in nums:
            found = False
            # We need to find x such that x | (x + 1) == num
            # The maximum possible x is num - 1 since x + 1 <= num
            for x in range(num):
                if (x | (x + 1)) == num:
                    ans.append(x)
                    found = True
                    break
            if not found:
                ans.append(-1)
        return ans