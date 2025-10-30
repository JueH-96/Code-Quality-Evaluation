def solve():
    n = input()
    if len(n) == 1:
        print("Yes")
        return
    
    for i in range(len(n) - 1):
        if int(n[i]) <= int(n[i+1]):
            print("No")
            return
    
    print("Yes")

solve()