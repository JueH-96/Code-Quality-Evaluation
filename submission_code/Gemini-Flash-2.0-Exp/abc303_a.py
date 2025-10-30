def solve():
    n = int(input())
    s = input()
    t = input()

    def are_similar(x, y):
        if x == y:
            return True
        if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
            return True
        if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
            return True
        return False

    for i in range(n):
        if not are_similar(s[i], t[i]):
            print("No")
            return

    print("Yes")

solve()