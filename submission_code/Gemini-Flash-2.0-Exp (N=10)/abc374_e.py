def solve():
    n, x = map(int, input().split())
    processes = []
    for _ in range(n):
        processes.append(list(map(int, input().split())))

    def check(capacity):
        total_cost = 0
        for a, p, b, q in processes:
            min_cost = float('inf')
            for num_s in range(100001):
                prod_s = a * num_s
                if prod_s >= capacity:
                    min_cost = min(min_cost, num_s * p)
                    continue
                
                rem_prod = capacity - prod_s
                if rem_prod <= 0:
                  min_cost = min(min_cost, num_s * p)
                  continue
                
                num_t = (rem_prod + b - 1) // b
                min_cost = min(min_cost, num_s * p + num_t * q)
            total_cost += min_cost
        return total_cost <= x

    left, right = 0, 10**9 + 1
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