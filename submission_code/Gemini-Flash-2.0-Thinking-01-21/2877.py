class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1, s2):
            if s2 in s1:
                return s1
            if s1 in s2:
                return s2
            n1, n2 = len(s1), len(s2)
            max_overlap = 0
            for i in range(1, min(n1, n2) + 1):
                if s1[n1 - i:] == s2[:i]:
                    max_overlap = i
            if max_overlap > 0:
                return s1 + s2[max_overlap:]
            else:
                return s1 + s2

        from itertools import permutations
        strs = [a, b, c]
        perms = list(permutations(strs))
        res = a + b + c + " " # Initialize with a string longer than any possible answer and lexicographically large

        for p in perms:
            merged_ab = merge(p[0], p[1])
            merged_abc = merge(merged_ab, p[2])
            if len(merged_abc) < len(res):
                res = merged_abc
            elif len(merged_abc) == len(res) and merged_abc < res:
                res = merged_abc
        return res