class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        result = []
        for s in arr:
            substrs = set()
            n = len(s)
            for i in range(n):
                for j in range(i, n):
                    substrs.add(s[i:j+1])
            sorted_subs = sorted(substrs, key=lambda x: (len(x), x))
            ans = ""
            for sub in sorted_subs:
                found = False
                for s2 in arr:
                    if s2 != s and sub in s2:
                        found = True
                        break
                if not found:
                    ans = sub
                    break
            result.append(ans)
        return result