def solve():
    n, x, y, z = map(int, input().split())
    if x < y:
        if x < z and z <= y:
            print("Yes")
        else:
            print("No")
    elif x > y:
        if y <= z and z <= x - 1:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    solve()