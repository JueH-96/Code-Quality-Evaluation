class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        def remove_last_char(s):
            return s[:-1]

        def can_be_equal(s1, s2, s3):
            while len(s1) > 0 and len(s2) > 0 and len(s3) > 0:
                if s1[-1] == s2[-1] == s3[-1]:
                    s1 = remove_last_char(s1)
                    s2 = remove_last_char(s2)
                    s3 = remove_last_char(s3)
                else:
                    break
            return s1, s2, s3

        s1, s2, s3 = can_be_equal(s1, s2, s3)

        if s1 == s2 == s3:
            return len(s1) + len(s2) + len(s3)
        else:
            return -1