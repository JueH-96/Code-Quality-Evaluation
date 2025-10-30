class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for x in nums:
            if x % 2 == 0:
                ans.append(-1)
            else:
                L = 0
                t = x
                while t & 1:
                    L += 1
                    t >>= 1
                A = x >> L
                lower_val = (1 << (L - 1)) - 1
                a_val = (A << L) | lower_val
                ans.append(a_val)
        return ans