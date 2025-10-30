# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))

    def get_dist(u, v, closed_edge):
        if u == v:
            return 0
        
        if closed_edge == n:
            if abs(u - v) == 1 or (u == 1 and v == n) or (u == n and v == 1):
                return float('inf')
            else:
                return min(abs(u - v), n - abs(u - v))
        else:
            if (u == closed_edge and v == closed_edge + 1) or (u == closed_edge + 1 and v == closed_edge):
                return float('inf')
            elif (u == n and v == 1 and closed_edge != n) or (u == 1 and v == n and closed_edge != n):
                return 1
            else:
                d1 = 0
                if u <= v:
                    if closed_edge >= u and closed_edge < v:
                        d1 = float('inf')
                    else:
                        d1 = v - u
                else:
                    if closed_edge >= v and closed_edge < u:
                        d1 = float('inf')
                    else:
                        d1 = n - (u - v)
                
                d2 = 0
                if u <= v:
                    if closed_edge >= v or closed_edge < u:
                        d2 = float('inf')
                    else:
                        d2 = n - (v - u)
                else:
                    if closed_edge >= u or closed_edge < v:
                        d2 = float('inf')
                    else:
                        d2 = u - v
                
                return min(d1, d2)

    ans = float('inf')
    for closed_edge in range(1, n + 1):
        total_dist = 0
        for i in range(m - 1):
            dist = get_dist(x[i], x[i+1], closed_edge)
            if dist == float('inf'):
                total_dist = float('inf')
                break
            total_dist += dist
        ans = min(ans, total_dist)

    print(ans)

solve()