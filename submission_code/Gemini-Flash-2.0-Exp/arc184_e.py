def solve():
    n, m = map(int, input().split())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    def transform(arr, x):
        temp_arr = [row[:] for row in arr]
        for _ in range(x):
            new_arr = [0] * m
            for k in range(m):
                s = 0
                for l in range(k + 1):
                    s += temp_arr[l]
                new_arr[k] = s % 2
            temp_arr = new_arr
        return temp_arr

    def f(i, j):
        arr1 = a[i-1]
        arr2 = a[j-1]
        
        if arr1 == arr2:
            return 0
        
        for x in range(1, n * m + 1):
            transformed_arr1 = transform(arr1, x)
            if transformed_arr1 == arr2:
                return x
        return 0

    total_sum = 0
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            total_sum = (total_sum + f(i, j)) % 998244353

    print(total_sum)

solve()