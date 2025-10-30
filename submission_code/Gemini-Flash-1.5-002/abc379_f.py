def solve():
    n, q = map(int, input().split())
    h = list(map(int, input().split()))

    def can_see(i, j):
        for k in range(i + 1, j):
            if h[k - 1] > h[j - 1]:
                return False
        return True

    for _ in range(q):
        l, r = map(int, input().split())
        count = 0
        for k in range(r + 1, n + 1):
            if can_see(l, k) and can_see(r, k):
                count += 1
        print(count)

solve()