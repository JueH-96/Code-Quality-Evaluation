def solve():
    n, m = map(int, input().split())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    def transform(arr):
        res = []
        for i in range(len(arr)):
            s = 0
            for j in range(i + 1):
                s += arr[j]
            res.append(s % 2)
        return res

    def f(i, j):
        arr1 = a[i]
        arr2 = a[j]
        for x in range(m + 1):
            if arr1 == arr2:
                return x
            arr1 = transform(arr1)
        return 0

    ans = 0
    for i in range(n):
        for j in range(i, n):
            ans = (ans + f(i, j)) % 998244353

    print(ans)

solve()