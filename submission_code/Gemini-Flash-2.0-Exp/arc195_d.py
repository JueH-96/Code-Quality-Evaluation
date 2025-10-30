def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def get_min_ops(arr):
        if not arr:
            return 0

        min_ops = float('inf')

        # Option 1: Swap adjacent elements
        for i in range(len(arr) - 1):
            new_arr = arr[:]
            new_arr[i], new_arr[i+1] = new_arr[i+1], new_arr[i]
            min_ops = min(min_ops, 1 + get_min_ops(new_arr))

        # Option 2: Delete prefix of equal elements
        for i in range(1, len(arr) + 1):
            if all(arr[j] == arr[0] for j in range(i)):
                min_ops = min(min_ops, 1 + get_min_ops(arr[i:]))

        return min_ops

    print(get_min_ops(a))

t = int(input())
for _ in range(t):
    solve()