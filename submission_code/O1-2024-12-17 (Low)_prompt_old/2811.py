class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        chosen = []
        i = 1
        while len(chosen) < n:
            if (k - i) not in chosen or (k - i) == i:
                chosen.append(i)
            i += 1
        return sum(chosen)