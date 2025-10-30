def solve():
    k = int(input())
    s = input()
    t = input()
    n = len(s)
    m = len(t)
    
    if abs(n - m) > k:
        print("No")
        return
    
    if s == t:
        print("Yes")
        return
    
    if n == m:
        diff = 0
        for i in range(n):
            if s[i] != t[i]:
                diff += 1
        if diff <= k:
            print("Yes")
        else:
            print("No")
    elif n > m:
        diff = 0
        i = 0
        j = 0
        while i < n and j < m:
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                diff += 1
                i += 1
        diff += (n - i)
        
        if diff <= k:
            print("Yes")
        else:
            print("No")
    else:
        diff = 0
        i = 0
        j = 0
        while i < n and j < m:
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                diff += 1
                j += 1
        diff += (m - j)
        if diff <= k:
            print("Yes")
        else:
            print("No")

solve()