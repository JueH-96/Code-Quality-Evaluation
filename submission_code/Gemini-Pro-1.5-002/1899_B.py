# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    ans = 0
    for k in range(1, n + 1):
        if n % k == 0:
            num_trucks = n // k
            truck_sums = []
            for i in range(num_trucks):
                current_sum = 0
                for j in range(k):
                    current_sum += a[i * k + j]
                truck_sums.append(current_sum)
            
            if len(truck_sums) > 1:
                ans = max(ans, abs(max(truck_sums) - min(truck_sums)))
            elif len(truck_sums) == 1:
                ans = max(ans, 0)

    print(ans)


t = int(input())
for _ in range(t):
    solve()