def solve():
    n, m = map(int, input().split())
    s = list(input())
    c = list(map(int, input().split()))

    for i in range(1, m + 1):
        indices = [j for j in range(n) if c[j] == i]
        if len(indices) > 0:
            last_char = s[indices[-1]]
            for j in range(len(indices) - 1, 0, -1):
                s[indices[j]] = s[indices[j-1]]
            s[indices[0]] = last_char
    print("".join(s))

solve()