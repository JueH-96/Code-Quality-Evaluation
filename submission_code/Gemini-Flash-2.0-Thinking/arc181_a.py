def solve():
    N = int(input())
    P = list(map(int, input().split()))

    def is_sorted(arr):
        for i in range(len(arr)):
            if arr[i] != i + 1:
                return False
        return True

    if is_sorted(P):
        print(0)
        return

    def apply_operation(arr, k):
        temp_arr = list(arr)
        if k >= 2:
            temp_arr[:k-1] = sorted(temp_arr[:k-1])
        if k <= N - 1:
            temp_arr[k:] = sorted(temp_arr[k:])
        return temp_arr

    for k in range(1, N + 1):
        operated_p = apply_operation(P, k)
        if is_sorted(operated_p):
            print(1)
            return

    print(2)

T = int(input())
for _ in range(T):
    solve()