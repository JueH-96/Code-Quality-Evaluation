class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations

        def merge2(s1, s2):
            # If one string is already contained in the other, return the larger one
            if s2 in s1:
                return s1
            if s1 in s2:
                return s2

            # Find maximum overlap of s1's suffix with s2's prefix
            max_overlap = 0
            limit = min(len(s1), len(s2))
            for i in range(1, limit + 1):
                if s1.endswith(s2[:i]):
                    max_overlap = i

            # Merge based on the overlap
            return s1 + s2[max_overlap:]

        best_answer = None

        # Try all permutations of the three strings
        for x, y, z in permutations([a, b, c]):
            merged_xy = merge2(x, y)
            merged_xyz = merge2(merged_xy, z)
            if best_answer is None:
                best_answer = merged_xyz
            else:
                # Pick shorter result or lexicographically smaller if the same length
                if len(merged_xyz) < len(best_answer):
                    best_answer = merged_xyz
                elif len(merged_xyz) == len(best_answer) and merged_xyz < best_answer:
                    best_answer = merged_xyz

        return best_answer