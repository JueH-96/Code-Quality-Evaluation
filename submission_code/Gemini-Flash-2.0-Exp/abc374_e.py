def solve():
    n, x = map(int, input().split())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    def check(capacity):
        cost = 0
        for i in range(n):
            ai, pi, bi, qi = a[i]
            min_cost = float('inf')
            for num_s in range(101):
                rem_prod = capacity - num_s * ai
                if rem_prod <= 0:
                    min_cost = min(min_cost, num_s * pi)
                    continue
                
                num_t = (rem_prod + bi - 1) // bi
                
                if num_s >= 0 and num_t >= 0:
                    min_cost = min(min_cost, num_s * pi + num_t * qi)
            
            if min_cost == float('inf'):
                return False
            cost += min_cost
        
        return cost <= x

    low = 0
    high = 10**9
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1

    print(ans)

solve()