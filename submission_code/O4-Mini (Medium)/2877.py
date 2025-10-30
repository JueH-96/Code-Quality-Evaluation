import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        # helper to merge two strings into the shortest common superstring
        def merge(s: str, t: str) -> str:
            # if one contains the other, return the longer
            if t in s:
                return s
            if s in t:
                return t
            # overlap s suffix with t prefix
            max_k1 = 0
            max_ov = min(len(s), len(t))
            for k in range(max_ov, 0, -1):
                if s[-k:] == t[:k]:
                    max_k1 = k
                    break
            s1 = s + t[max_k1:]
            # overlap t suffix with s prefix
            max_k2 = 0
            for k in range(max_ov, 0, -1):
                if t[-k:] == s[:k]:
                    max_k2 = k
                    break
            s2 = t + s[max_k2:]
            # choose the shorter, or lexicographically smaller if tie
            if len(s1) < len(s2):
                return s1
            if len(s2) < len(s1):
                return s2
            return min(s1, s2)
        
        best = None
        # try all orders of merging a, b, c
        for x, y, z in itertools.permutations([a, b, c], 3):
            merged_xy = merge(x, y)
            merged_xyz = merge(merged_xy, z)
            if best is None:
                best = merged_xyz
            else:
                if len(merged_xyz) < len(best):
                    best = merged_xyz
                elif len(merged_xyz) == len(best) and merged_xyz < best:
                    best = merged_xyz
        return best