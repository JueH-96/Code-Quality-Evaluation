class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1: str, s2: str) -> str:
            if s2 in s1:
                return s1
            for i in range(min(len(s1), len(s2)), 0, -1):
                if s1[-i:] == s2[:i]:
                    return s1 + s2[i:]
            return s1 + s2

        def solve(s1: str, s2: str, s3: str) -> str:
            return merge(merge(s1, s2), s3)

        ans = ""
        for s in [
            solve(a, b, c),
            solve(a, c, b),
            solve(b, a, c),
            solve(b, c, a),
            solve(c, a, b),
            solve(c, b, a),
        ]:
            if not ans or len(s) < len(ans) or (len(s) == len(ans) and s < ans):
                ans = s
        return ans