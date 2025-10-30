def solve():
    n = int(input())
    s = input()

    if n % 2 == 0:
        print("No")
        return

    mid = (n + 1) // 2
    
    if s[mid - 1] != '/':
        print("No")
        return

    for i in range(mid - 1):
        if s[i] != '1':
            print("No")
            return

    for i in range(mid, n):
        if s[i] != '2':
            print("No")
            return

    print("Yes")

solve()