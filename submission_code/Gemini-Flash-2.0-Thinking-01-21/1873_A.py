def solve():
    s = input()
    if s == "abc":
        print("YES")
        return
    
    l = list(s)
    l[0], l[1] = l[1], l[0]
    s1 = "".join(l)
    if s1 == "abc":
        print("YES")
        return
        
    l = list(s)
    l[0], l[2] = l[2], l[0]
    s2 = "".join(l)
    if s2 == "abc":
        print("YES")
        return
        
    l = list(s)
    l[1], l[2] = l[2], l[1]
    s3 = "".join(l)
    if s3 == "abc":
        print("YES")
        return
        
    print("NO")

t = int(input())
for _ in range(t):
    solve()