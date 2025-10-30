class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        ans = n

        for i in range(1 << n):
            temp = bin(i)[2:].zfill(n)
            cnt = 0
            t = ""
            for j in range(n):
                if temp[j] == '1':
                    cnt += 1
                    if s[j] == '0':
                        t += '1'
                    else:
                        t += '0'
                else:
                    t += s[j]

            if cnt > numOps:
                continue

            mx = 0
            cur = 0
            if n > 0:
                cur = 1
                mx = 1
            
            for j in range(1,n):
                if t[j] == t[j-1]:
                    cur += 1
                else:
                    cur = 1
                mx = max(mx, cur)
            
            ans = min(ans, mx)
        
        return ans