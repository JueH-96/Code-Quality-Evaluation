class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        a, b, c = nums
        perm1 = int(bin(a)[2:] + bin(b)[2:] + bin(c)[2:], 2)
        perm2 = int(bin(a)[2:] + bin(c)[2:] + bin(b)[2:], 2)
        perm3 = int(bin(b)[2:] + bin(a)[2:] + bin(c)[2:], 2)
        perm4 = int(bin(b)[2:] + bin(c)[2:] + bin(a)[2:], 2)
        perm5 = int(bin(c)[2:] + bin(a)[2:] + bin(b)[2:], 2)
        perm6 = int(bin(c)[2:] + bin(b)[2:] + bin(a)[2:], 2)
        return max(perm1, perm2, perm3, perm4, perm5, perm6)