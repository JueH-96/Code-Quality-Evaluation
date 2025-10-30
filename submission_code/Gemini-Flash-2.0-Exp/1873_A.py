def solve():
    s = input()
    if s == "abc":
        print("YES")
        return
    
    for i in range(3):
        for j in range(i + 1, 3):
            temp = list(s)
            temp[i], temp[j] = temp[j], temp[i]
            if "".join(temp) == "abc":
                print("YES")
                return
    
    print("NO")

t = int(input())
for _ in range(t):
    solve()