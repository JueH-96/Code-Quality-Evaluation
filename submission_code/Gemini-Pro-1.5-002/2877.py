class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def solve(p, q, r):
            ans = p + q + r
            for i in range(1, len(p) + 1):
                if q.startswith(p[-i:]):
                    ans = min(ans, p + q[i:], key=len)
                    ans = min(ans, p + q[i:] + r, key=len)
                    if r.startswith(q[i:]):
                        ans = min(ans, p + r, key=len)
            for i in range(1, len(q) + 1):
                if r.startswith(q[-i:]):
                    ans = min(ans, p + q + r[i:], key=len)
            
            return ans

        ans = ""
        ans = min(ans, solve(a, b, c), key=len)
        ans = min(ans, solve(a, c, b), key=len)
        ans = min(ans, solve(b, a, c), key=len)
        ans = min(ans, solve(b, c, a), key=len)
        ans = min(ans, solve(c, a, b), key=len)
        ans = min(ans, solve(c, b, a), key=len)
        
        return min(ans, key = lambda x: (len(x), x))