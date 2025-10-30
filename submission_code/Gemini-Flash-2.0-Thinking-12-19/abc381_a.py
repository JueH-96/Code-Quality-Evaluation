def solve():
    n = int(input())
    s = input()
    if n % 2 == 0:
        print("No")
        return
    m = (n + 1) // 2
    for i in range(m - 1):
        if s[i] != '1':
            print("No")
            return
    if s[m - 1] != '/':
        print("No")
        return
    for i in range(m, n):
        if s[i] != '2':
            print("No")
            return
    print("Yes")

if __name__ == '__main__':
    solve()