class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        @cache
        def solve(index, factor):
            if index == len(strength):
                return 0
            lo = 0
            hi = factor*2
            while lo < hi:
                time = lo + (hi - lo) // 2
                energy = factor * time + (time * (time + 1)) / 2
                if energy >= strength[index]:
                    hi = time
                else:
                    lo = time + 1
            return lo + solve(index + 1, factor + K)
        return solve(0, 1)