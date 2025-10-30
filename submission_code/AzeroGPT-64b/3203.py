class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        def preprocess(s):
            n = len(s)
            cnt = [collections.defaultdict(int) for _ in range(n+1)]
            for i in range(n):
                cnt[i+1] = cnt[i].copy()
                cnt[i+1][s[i]] += 1
            return cnt
        
        cnt = preprocess(s)
        res = []
        for a, b, c, d in queries:
            d1, d2 = {}, {}
            for c1, v1 in cnt[b+1].items():
                d1[c1] = v1 - (cnt[a][c1] if a else 0)
            for c1, v1 in cnt[d+1].items():
                d2[c1] = v1 - (cnt[c][c1] if c else 0)
            mv = -100
            for k1, v1 in d1.items():
                if v1 >= mv:
                    mv = v1
                    c2 = k1
                elif v1 > mv // 2:
                    d2[k1] -= v1
            for k2, v2 in d2.items():
                if k2 == c2:
                    if mv % 2 and v2 >= mv // 2 + 1:
                        res.append(True)
                        break
                    elif not mv % 2 and v2 > mv // 2:
                        res.append(True)
                        break
                elif v2 >= mv // 2:
                    res.append(True)
                    break
            else:
                res.append(False)
        return res