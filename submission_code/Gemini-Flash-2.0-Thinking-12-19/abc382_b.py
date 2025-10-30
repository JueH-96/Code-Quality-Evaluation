def solve():
    n, d = map(int, input().split())
    s_list = list(input())

    for _ in range(d):
        for i in range(n - 1, -1, -1):
            if s_list[i] == '@':
                s_list[i] = '.'
                break

    print("".join(s_list))

if __name__ == '__main__':
    solve()