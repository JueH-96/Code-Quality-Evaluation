class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        import itertools

        def merge(s1, s2):
            if s2 in s1:
                return s1
            if s1 in s2:
                return s2

            n1, n2 = len(s1), len(s2)
            max_overlap = 0

            # Check overlap of s1 suffix with s2 prefix
            for i in range(1, min(n1, n2) + 1):
                if s1[n1 - i:] == s2[:i]:
                    max_overlap = max(max_overlap, i)

            if max_overlap > 0:
                return s1 + s2[max_overlap:]

            return s1 + s2

        strings = [a, b, c]
        permutations = list(itertools.permutations(strings))

        min_len = float('inf')
        result = "~"  # Initialize with a lexicographically large string

        for p in permutations:
            merged1 = merge(p[0], p[1])
            merged2 = merge(merged1, p[2])

            contains_a = a in merged2
            contains_b = b in merged2
            contains_c = c in merged2

            if contains_a and contains_b and contains_c:
                if len(merged2) < min_len:
                    min_len = len(merged2)
                    result = merged2
                elif len(merged2) == min_len:
                    if merged2 < result:
                        result = merged2

        return result