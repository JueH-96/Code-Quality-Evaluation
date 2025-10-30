class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def solve(s1, s2, s3):
            def merge(s1, s2):
                if s1 in s2:
                    return s2
                if s2 in s1:
                    return s1
                for i in range(1, min(len(s1), len(s2)) + 1):
                    if s1[-i:] == s2[:i]:
                        return s1 + s2[i:]
                return s1 + s2

            res12 = merge(s1, s2)
            res123 = merge(res12, s3)
            res21 = merge(s2, s1)
            res213 = merge(res21, s3)
            res13 = merge(s1, s3)
            res132 = merge(res13, s2)
            res31 = merge(s3, s1)
            res312 = merge(res31, s2)
            res23 = merge(s2, s3)
            res231 = merge(res23, s1)
            res32 = merge(s3, s2)
            res321 = merge(res32, s1)
            
            res = [res123, res213, res132, res312, res231, res321]
            min_len = float('inf')
            ans = ""
            for s in res:
                if len(s) < min_len:
                    min_len = len(s)
                    ans = s
                elif len(s) == min_len and s < ans:
                    ans = s
            return ans

        return solve(a, b, c)