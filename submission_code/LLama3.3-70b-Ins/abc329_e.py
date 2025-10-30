def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()

    x = ['#'] * n

    while True:
        changed = False
        for i in range(n - m + 1):
            if ''.join(x[i:i+m]) == '#' * m:
                x[i:i+m] = list(t)
                changed = True

        if not changed:
            break

    if ''.join(x) == s:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()