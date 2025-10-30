def solve():
    n, m = map(int, input().split())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    def transform(arr):
        new_arr = []
        cur_sum = 0
        for x in arr:
            cur_sum += x
            new_arr.append(cur_sum % 2)
        return new_arr

    def f(i, j):
        arr_i = a[i-1][:]
        arr_j = a[j-1][:]
        
        x = 0
        while x <= m:
            if arr_i == arr_j:
                return x
            arr_i = transform(arr_i)
            x += 1
        return 0

    total_sum = 0
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            total_sum = (total_sum + f(i, j)) % 998244353

    print(total_sum)

solve()