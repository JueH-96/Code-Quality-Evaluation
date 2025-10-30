def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_distinct_sum = 0
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            sub1 = a[:i]
            sub2 = a[i:j]
            sub3 = a[j:]

            distinct_count1 = len(set(sub1))
            distinct_count2 = len(set(sub2))
            distinct_count3 = len(set(sub3))

            max_distinct_sum = max(max_distinct_sum, distinct_count1 + distinct_count2 + distinct_count3)

    print(max_distinct_sum)

solve()