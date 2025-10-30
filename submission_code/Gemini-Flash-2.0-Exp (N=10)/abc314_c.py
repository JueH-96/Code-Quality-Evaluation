def solve():
    n, m = map(int, input().split())
    s = list(input())
    colors = list(map(int, input().split()))

    for color in range(1, m + 1):
        indices = [i for i, c in enumerate(colors) if c == color]
        if len(indices) > 1:
            last_char = s[indices[-1]]
            for i in range(len(indices) - 1, 0, -1):
                s[indices[i]] = s[indices[i-1]]
            s[indices[0]] = last_char
    print("".join(s))

solve()