def solve():
    n = int(input())
    a = list(map(int, input().split()))

    results = []
    for i in range(n):
        current_val = a[i]
        current_sum = 0
        for j in range(n):
            if a[j] > current_val:
                current_sum += a[j]
        results.append(current_sum)

    print(*results)

solve()