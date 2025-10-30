class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for prime in nums:
            found = False
            # We'll search x in [0..prime-1] to find the smallest x with x OR (x+1) == prime
            for x in range(prime):
                if (x | (x + 1)) == prime:
                    ans.append(x)
                    found = True
                    break
            if not found:
                ans.append(-1)
        return ans