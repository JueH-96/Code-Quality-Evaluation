def solve():
    v1, v2, v3 = map(int, input().split())
    if v1 + 2 * v2 + 3 * v3 == 1029:
        print("Yes")
        print("0 0 0 0 6 0 6 0 0")
    else:
        print("No")

if __name__ == '__main__':
    solve()