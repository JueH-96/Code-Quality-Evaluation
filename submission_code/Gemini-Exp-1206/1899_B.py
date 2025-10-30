def solve():
    n = int(input())
    a = list(map(int, input().split()))

    ans = 0
    for k in range(1, n + 1):
        if n % k == 0:
            truck_sums = []
            for i in range(0, n, k):
                truck_sums.append(sum(a[i:i+k]))
            
            if len(truck_sums) > 1:
                ans = max(ans, max(truck_sums) - min(truck_sums))

    print(ans)

t = int(input())
for _ in range(t):
    solve()