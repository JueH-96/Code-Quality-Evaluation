def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))

    def calculate_mex(arr):
        s = set(arr)
        mex = 0
        while mex in s:
            mex += 1
        return mex

    for _ in range(q):
        i, x = map(int, input().split())
        a[i - 1] = x
        print(calculate_mex(a))

solve()