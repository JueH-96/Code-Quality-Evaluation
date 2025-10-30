class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        chosen = []
        current = 1
        while len(chosen) < n:
            if (k - current) not in chosen:
                chosen.append(current)
            current += 1
        return sum(chosen)