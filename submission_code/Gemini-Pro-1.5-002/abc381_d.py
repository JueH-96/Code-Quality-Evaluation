def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def is_1122(arr):
        if len(arr) % 2 != 0:
            return False
        for i in range(0, len(arr), 2):
            if arr[i] != arr[i+1]:
                return False
        counts = {}
        for x in arr:
            counts[x] = counts.get(x, 0) + 1
        for x in counts:
            if counts[x] != 2 and counts[x] != 0:
                return False
        return True

    max_len = 0
    for i in range(n):
        for j in range(i, n):
            subarray = a[i:j+1]
            if is_1122(subarray):
                max_len = max(max_len, len(subarray))

    print(max_len)

solve()