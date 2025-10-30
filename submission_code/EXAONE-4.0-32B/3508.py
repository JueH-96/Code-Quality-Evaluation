class Solution:
    def minChanges(self, n: int, k: int) -> int:
        count = 0
        for i in range(22):
            kb = (k >> i) & 1
            nb = (n >> i) & 1
            if kb == 1:
                if nb == 0:
                    return -1
            else:
                if nb == 1:
                    count += 1
        return count