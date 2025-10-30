class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations

        # Function to calculate the maximum overlap between s1 and s2
        # where s1 is the prefix and s2 is the suffix in the merged string
        def overlap(s1, s2):
            max_len = 0
            min_len = min(len(s1), len(s2))
            for i in range(1, min_len + 1):
                if s1[-i:] == s2[:i]:
                    max_len = i
            return max_len
        
        # Merges s1 and s2 into the shortest possible string that contains both
        def merge_strings(s1, s2):
            # If one string is already in the other, no merge needed
            if s1 in s2:
                return s2
            if s2 in s1:
                return s1
            # Otherwise, merge using maximal overlap
            ol = overlap(s1, s2)
            return s1 + s2[ol:]

        # Try all permutations of (a, b, c)
        candidates = []
        for x, y, z in permutations([a, b, c]):
            merged_xy = merge_strings(x, y)
            merged_xyz = merge_strings(merged_xy, z)
            candidates.append(merged_xyz)

        # Among all possible candidates, return the one with minimal length,
        # and if there's a tie, the lexicographically smallest
        ans = min(candidates, key=lambda s: (len(s), s))
        return ans