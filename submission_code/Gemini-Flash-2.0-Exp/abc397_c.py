def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_distinct_sum = 0
    for i in range(1, n):
        left_subarray = a[:i]
        right_subarray = a[i:]

        distinct_left = len(set(left_subarray))
        distinct_right = len(set(right_subarray))

        max_distinct_sum = max(max_distinct_sum, distinct_left + distinct_right)

    print(max_distinct_sum)

solve()