def solve():
    n, m = map(int, input().split())
    s = list(input())
    c = list(map(int, input().split()))

    # Group characters by color
    color_chars = [[] for _ in range(m + 1)]
    for i in range(n):
        color_chars[c[i]].append(s[i])

    # Perform right circular shift for each color
    for i in range(1, m + 1):
        color_chars[i] = color_chars[i][-1:] + color_chars[i][:-1]

    # Reconstruct the string
    result = [''] * n
    for i in range(n):
        result[i] = color_chars[c[i]].pop(0)

    print(''.join(result))

if __name__ == '__main__':
    solve()