def solve():
    s = input()
    if s == "abc":
        print("YES")
        return
    
    if s[0] == 'a' and s[1] == 'c' and s[2] == 'b':
        print("YES")
        return
    
    if s[0] == 'b' and s[1] == 'a' and s[2] == 'c':
        print("YES")
        return
    
    if s[0] == 'c' and s[1] == 'b' and s[2] == 'a':
        print("YES")
        return
    
    print("NO")

t = int(input())
for _ in range(t):
    solve()