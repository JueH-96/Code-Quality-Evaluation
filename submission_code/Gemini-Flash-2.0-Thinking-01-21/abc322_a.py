def solve():
    n = int(input())
    s = input()

    for i in range(n - 2):
        if s[i:i+3] == "ABC":
            print(i + 1)
            return

    print(-1)

if __name__ == "__main__":
    solve()