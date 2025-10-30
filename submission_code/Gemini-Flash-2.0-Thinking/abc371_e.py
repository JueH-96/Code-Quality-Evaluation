def solve():
    n = int(input())
    a = list(map(int, input().split()))

    total_sum = 0
    last_occurrence = {}

    for k in range(n):
        val = a[k]
        p = last_occurrence.get(val, -1)

        num_start = k - p
        num_end = n - k

        contribution = num_start * num_end
        total_sum += contribution
        last_occurrence[val] = k

    print(total_sum)

solve()