class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        min_len = min(len(s1), len(s2), len(s3))
        len_common = 0
        while len_common < min_len and s1[len_common] == s2[len_common] and s2[len_common] == s3[len_common]:
            len_common += 1
        if len_common == 0:
            return -1
        else:
            total_len = len(s1) + len(s2) + len(s3)
            return total_len - 3 * len_common