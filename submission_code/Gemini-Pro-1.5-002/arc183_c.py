# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    conditions = []
    for _ in range(m):
        l, r, x = map(int, input().split())
        conditions.append((l - 1, r - 1, x - 1))

    ans = 0
    import itertools
    for p in itertools.permutations(range(n)):
        valid = True
        for l, r, x in conditions:
            mx = -1
            mx_idx = -1
            for i in range(l, r + 1):
                if p[i] > mx:
                    mx = p[i]
                    mx_idx = i
            if mx_idx == x:
                valid = False
                break
        if valid:
            ans = (ans + 1) % 998244353

    print(ans)

solve()