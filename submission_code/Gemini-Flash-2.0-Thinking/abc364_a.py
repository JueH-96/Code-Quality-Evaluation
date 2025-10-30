def solve():
    n = int(input())
    s = [input() for _ in range(n)]

    for i in range(n - 1):
        if s[i] == 'sweet' and s[i+1] == 'sweet':
            print("No")
            return

    print("Yes")

solve()