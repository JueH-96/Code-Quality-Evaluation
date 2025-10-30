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

    for op_idx in range(m):
        for k in range(2, a[op_idx] + 1):
            for i in range(k - 1):
                if p[i] > p[i+1]:
                    p[i], p[i+1] = p[i+1], p[i]
        print(calculate_inversions(p))

solve()