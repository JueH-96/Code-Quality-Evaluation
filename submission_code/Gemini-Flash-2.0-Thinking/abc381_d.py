def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def is_1122(arr):
        length = len(arr)
        if length % 2 != 0:
            return False

        for i in range(0, length // 2):
            if arr[2 * i] != arr[2 * i + 1]:
                return False

        counts = {}
        for x in arr:
            counts[x] = counts.get(x, 0) + 1

        for count in counts.values():
            if count != 2:
                return False

        return True

    max_length = 0
    for i in range(n):
        for j in range(i, n):
            subarray = a[i : j + 1]
            if is_1122(subarray):
                max_length = max(max_length, len(subarray))

    print(max_length)

solve()