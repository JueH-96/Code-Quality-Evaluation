def solve():
    n, m = map(int, input().split())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    def transform(arr):
        new_arr = []
        current_sum = 0
        for x in arr:
            current_sum = (current_sum + x) % 2
            new_arr.append(current_sum)
        return new_arr

    def calculate_f(arr1, arr2):
        if arr1 == arr2:
            return 0
        
        temp_arr1 = arr1[:]
        for steps in range(1, m + 2):
            temp_arr1 = transform(temp_arr1)
            if temp_arr1 == arr2:
                return steps
        return 0

    total_sum = 0
    for i in range(n):
        for j in range(i, n):
            total_sum = (total_sum + calculate_f(a[i], a[j])) % 998244353
    print(total_sum)

solve()