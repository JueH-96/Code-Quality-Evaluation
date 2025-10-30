class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations

        def merge(x: str, y: str) -> str:
            # If one string already contains the other, return the longer one
            if y in x:
                return x
            if x in y:
                return y
            # Otherwise find the maximum overlap of suffix of x with prefix of y
            max_ov = 0
            max_len = min(len(x), len(y))
            for k in range(1, max_len + 1):
                if x.endswith(y[:k]):
                    max_ov = k
            # Merge with the maximal overlap
            return x + y[max_ov:]

        best = None
        # Try all orders of merging the three strings
        for x, y, z in permutations([a, b, c], 3):
            merged_xy = merge(x, y)
            merged_xyz = merge(merged_xy, z)
            # Keep the shortest; if tie, the lexicographically smallest
            if (best is None or
                len(merged_xyz) < len(best) or
                (len(merged_xyz) == len(best) and merged_xyz < best)):
                best = merged_xyz

        return best