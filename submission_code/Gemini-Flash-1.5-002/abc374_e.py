def solve():
    n, x = map(int, input().split())
    processes = []
    for _ in range(n):
        processes.append(list(map(int, input().split())))

    def check(capacity):
        costs = []
        for a, p, b, q in processes:
            min_cost = float('inf')
            
            #Check if possible with only S
            if a > 0:
                num_s = (capacity + a -1 ) // a
                min_cost = min(min_cost, num_s * p)
            
            #Check if possible with only T
            if b > 0:
                num_t = (capacity + b - 1) // b
                min_cost = min(min_cost, num_t * q)
            
            #Check if impossible
            if min_cost == float('inf'):
                return False
            costs.append(min_cost)
        return sum(costs) <= x

    left = 0
    right = 10**9
    ans = 0
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    print(ans)

solve()