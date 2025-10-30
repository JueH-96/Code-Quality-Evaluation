class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def get_overlap_len(s1, s2):
            for length in range(min(len(s1), len(s2)), -1, -1):
                if s1[-length:] == s2[:length]:
                    return length
            return 0

        def merge_strings(s1, s2):
            overlap_len = get_overlap_len(s1, s2)
            return s1 + s2[overlap_len:]

        import itertools
        strings = [a, b, c]
        permutations = list(itertools.permutations(strings))
        min_len = float('inf')
        min_str = ""

        for p in permutations:
            s1, s2, s3 = p
            merged12 = merge_strings(s1, s2)
            merged_all = merge_strings(merged12, s3)
            if len(merged_all) < min_len:
                min_len = len(merged_all)
                min_str = merged_all
            elif len(merged_all) == min_len:
                if min_str == "" or merged_all < min_str:
                    min_str = merged_all
        return min_str