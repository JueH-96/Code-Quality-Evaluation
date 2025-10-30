def solve():
    n = int(input())
    p = list(map(int, input().split()))
    m = int(input())
    a = list(map(int, input().split()))

    def calculate_inversions(arr):
        count = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    count += 1
        return count

    def apply_operation(arr, k):
        for i in range(k - 1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
        return arr

    original_p = p[:]
    for i in range(m):
        p = original_p[:]
        for j in range(i + 1):
            p = apply_operation(p, a[j])
        print(calculate_inversions(p))

solve()