def solve():
    n, q = map(int, input().split())
    s = list(input())

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            l, r = query[1] - 1, query[2] - 1
            for i in range(l, r + 1):
                s[i] = '1' if s[i] == '0' else '0'
        else:
            l, r = query[1] - 1, query[2] - 1
            is_good = True
            for i in range(l, r):
                if s[i] == s[i + 1]:
                    is_good = False
                    break
            print("Yes" if is_good else "No")

if __name__ == "__main__":
    solve()