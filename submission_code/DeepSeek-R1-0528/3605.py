class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for n in nums:
            candidate = -1
            for k in range(31, -1, -1):
                bit = 1 << k
                if bit > n:
                    continue
                mask = bit - 1
                if (n & mask) != mask:
                    continue
                A = n >> k
                if A & 1:
                    candidate = n - bit
                    break
            ans.append(candidate)
        return ans