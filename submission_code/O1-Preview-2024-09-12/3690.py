class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            if self.is_possible(s, numOps, mid):
                right = mid
            else:
                left = mid + 1
        return left

    def is_possible(self, s, numOps, L):
        n = len(s)
        flips_needed = self.flips_required(s, L)
        return flips_needed <= numOps

    def flips_required(self, s, L):
        n = len(s)
        flips_needed = 0
        for c in ['0', '1']:
            i = 0
            while i <= n - (L + 1):
                if all(s[j] == c for j in range(i, i + L + 1)):
                    flips_needed += 1
                    i += L + 1  # Skip over this segment
                else:
                    i += 1
        return flips_needed