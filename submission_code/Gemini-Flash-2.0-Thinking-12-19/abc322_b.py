def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()

    is_prefix = False
    is_suffix = False

    if len(t) >= n:
        if t[:n] == s:
            is_prefix = True
        if t[m-n:] == s:
            is_suffix = True

    if is_prefix and is_suffix:
        print(0)
    elif is_prefix:
        print(1)
    elif is_suffix:
        print(2)
    else:
        print(3)

if __name__ == "__main__":
    solve()