class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        ans = [""] * n

        for i in range(n):
            s = arr[i]
            m = len(s)
            found = False
            for length in range(1, m + 1):
                subs = []
                for j in range(m - length + 1):
                    subs.append(s[j:j+length])
                
                subs.sort()
                
                for sub in subs:
                    ok = True
                    for k in range(n):
                        if i != k and sub in arr[k]:
                            ok = False
                            break
                    if ok:
                        ans[i] = sub
                        found = True
                        break
                if found:
                    break
        return ans