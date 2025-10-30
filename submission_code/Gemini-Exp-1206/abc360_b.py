def solve():
    s = input()
    t = input()
    n = len(s)
    m = len(t)
    
    for w in range(1, n):
        for c in range(1, w + 1):
            res = ""
            for i in range(0, n, w):
                if i + w <= n:
                    if i + c -1 < i + w:
                        res += s[i + c - 1]
                elif i + c - 1 < n:
                    res += s[i + c - 1]
            
            if res == t:
                print("Yes")
                return
    
    print("No")

solve()