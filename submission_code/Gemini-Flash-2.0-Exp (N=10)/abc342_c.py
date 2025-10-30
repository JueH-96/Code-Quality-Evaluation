def solve():
    n = int(input())
    s = list(input())
    q = int(input())
    
    for _ in range(q):
        c, d = input().split()
        for i in range(len(s)):
            if s[i] == c:
                s[i] = d
    
    print("".join(s))

solve()