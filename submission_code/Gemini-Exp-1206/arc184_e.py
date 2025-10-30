def solve():
    n, m = map(int, input().split())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    def transform(arr):
        res = []
        curr_sum = 0
        for x in arr:
            curr_sum = (curr_sum + x) % 2
            res.append(curr_sum)
        return res

    def calculate_f(arr1, arr2):
        if arr1 == arr2:
            return 0
        
        for x in range(1, m + 2):
            arr1 = transform(arr1)
            if arr1 == arr2:
                return x
        return 0

    total_f = 0
    for i in range(n):
        for j in range(i, n):
            total_f = (total_f + calculate_f(a[i][:], a[j][:])) % 998244353

    print(total_f)

solve()