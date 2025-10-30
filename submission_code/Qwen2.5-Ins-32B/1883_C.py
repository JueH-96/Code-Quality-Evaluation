def min_operations_to_divide_by_k(n, k, arr):
    if k == 2:
        for i in range(n):
            if arr[i] % 2 == 0:
                return 0
        return 1
    elif k == 3:
        for i in range(n):
            if arr[i] % 3 == 0:
                return 0
        return 1
    elif k == 4:
        for i in range(n):
            if arr[i] % 4 == 0:
                return 0
        for i in range(n):
            if arr[i] % 2 == 0:
                return 1
        return 2
    elif k == 5:
        for i in range(n):
            if arr[i] % 5 == 0:
                return 0
        return 1

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    arr = list(map(int, input().split()))
    print(min_operations_to_divide_by_k(n, k, arr))