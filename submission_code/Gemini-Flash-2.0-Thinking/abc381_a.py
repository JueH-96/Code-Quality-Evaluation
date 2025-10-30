def solve():
    n = int(input())
    s = input()

    if n % 2 == 0:
        print("No")
        return

    mid_index = (n + 1) // 2

    if s[mid_index - 1] != '/':
        print("No")
        return

    for i in range(mid_index - 1):
        if s[i] != '1':
            print("No")
            return

    for i in range(mid_index, n):
        if s[i] != '2':
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    solve()