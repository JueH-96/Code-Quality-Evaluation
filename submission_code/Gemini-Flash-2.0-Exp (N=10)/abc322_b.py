def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()

    is_prefix = t.startswith(s)
    is_suffix = t.endswith(s)

    if is_prefix and is_suffix:
        print(0)
    elif is_prefix:
        print(1)
    elif is_suffix:
        print(2)
    else:
        print(3)

solve()