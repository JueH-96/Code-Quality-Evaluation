class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        chosen = set()
        ans = 0
        x = 1
        while len(chosen) < n:
            # if k - x is not in chosen, then x can be safely added
            # without forming a pair that sums to k
            if (k - x) not in chosen:
                chosen.add(x)
                ans += x
            x += 1
        return ans