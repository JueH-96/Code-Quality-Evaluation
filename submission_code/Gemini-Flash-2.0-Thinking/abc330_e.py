def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))

    def calculate_mex(arr):
        seen = set(arr)
        mex_val = 0
        while mex_val in seen:
            mex_val += 1
        return mex_val

    for _ in range(q):
        i, x = map(int, input().split())
        a[i - 1] = x
        print(calculate_mex(a))

solve()