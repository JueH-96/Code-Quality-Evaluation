def solve():
    n, k = map(int, input().split())
    s = input()
    q = s.count('?')
    ans = 0
    mod = 998244353

    for i in range(2**q):
        temp = bin(i)[2:].zfill(q)
        curr = 0
        t = ""
        for j in range(n):
            if s[j] == '?':
                t += 'A' if temp[curr] == '0' else 'B'
                curr += 1
            else:
                t += s[j]
        
        good = True
        for j in range(n - k + 1):
            sub = t[j:j+k]
            if sub == sub[::-1]:
                good = False
                break
        
        if good:
            ans = (ans + 1) % mod
    
    print(ans)

solve()