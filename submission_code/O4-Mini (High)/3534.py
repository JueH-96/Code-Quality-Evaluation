from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        # Helper to check if by at most one swap within a we can get b
        def can_swap_to(a: int, b: int) -> bool:
            if a == b:
                return True
            s = list(str(a))
            n = len(s)
            # Try all pairs of positions to swap
            for i in range(n):
                for j in range(i + 1, n):
                    # swap i, j
                    s[i], s[j] = s[j], s[i]
                    # check numeric value
                    if int("".join(s)) == b:
                        # restore and return
                        s[i], s[j] = s[j], s[i]
                        return True
                    # restore and continue
                    s[i], s[j] = s[j], s[i]
            return False

        ans = 0
        N = len(nums)
        # check all pairs i < j
        for i in range(N):
            for j in range(i + 1, N):
                x, y = nums[i], nums[j]
                # if one swap in x makes y, or one swap in y makes x
                if can_swap_to(x, y) or can_swap_to(y, x):
                    ans += 1
        return ans