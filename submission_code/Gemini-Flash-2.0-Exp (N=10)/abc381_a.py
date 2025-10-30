def solve():
    n = int(input())
    s = input()

    if n % 2 == 0:
        print("No")
        return
    
    mid = (n + 1) // 2
    
    for i in range(n):
        if i < mid - 1:
            if s[i] != '1':
                print("No")
                return
        elif i == mid - 1:
            if s[i] != '/':
                print("No")
                return
        else:
            if s[i] != '2':
                print("No")
                return
    
    print("Yes")

solve()