def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_sum = -float('inf')

    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            if j == i:
                current_sum = a[j]
                max_sum = max(max_sum, current_sum)
            elif a[j] % 2 != a[j - 1] % 2:
                current_sum += a[j]
                max_sum = max(max_sum, current_sum)
            else:
                break

    print(max_sum)

t = int(input())
for _ in range(t):
    solve()