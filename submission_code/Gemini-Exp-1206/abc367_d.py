def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    prefix_sums = [0] * (n + 1)
    for i in range(n):
        prefix_sums[i + 1] = prefix_sums[i] + a[i]

    count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            dist = 0
            if i < j:
                dist = prefix_sums[j] - prefix_sums[i]
            else:
                dist = prefix_sums[n] - prefix_sums[i] + prefix_sums[j]

            if dist % m == 0:
                count += 1

    print(count)

solve()