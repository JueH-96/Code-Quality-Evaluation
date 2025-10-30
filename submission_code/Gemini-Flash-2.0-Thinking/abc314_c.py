def solve():
    n, m = map(int, input().split())
    s = list(input())
    c = list(map(int, input().split()))

    for color in range(1, m + 1):
        indices = [i for i, col in enumerate(c) if col == color]
        if not indices:
            continue

        if len(indices) == 1:
            continue

        chars_with_color = [s[i] for i in indices]
        shifted_chars = [chars_with_color[-1]] + chars_with_color[:-1]

        for i in range(len(indices)):
            s[indices[i]] = shifted_chars[i]

    print("".join(s))

solve()