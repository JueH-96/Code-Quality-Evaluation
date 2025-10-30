class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        L = len(pattern)
        n = len(s)
        if L > n:
            return -1
        s_list = list(s)
        pat_list = list(pattern)
        for i in range(n - L + 1):
            count = 0
            for k in range(L):
                if s_list[i + k] != pat_list[k]:
                    count += 1
                    if count > 1:
                        break
            if count <= 1:
                return i
        return -1