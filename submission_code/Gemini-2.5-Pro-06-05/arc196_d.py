# YOUR CODE HERE
import sys

def solve():
    """
    This function reads input, solves the problem, and prints the output.
    """
    N, M, Q = map(int, sys.stdin.readline().split())
    
    journeys = []
    for _ in range(M):
        s, t = map(int, sys.stdin.readline().split())
        journeys.append((s, t))

    queries = []
    for i in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        queries.append((l-1, r-1, i))

    first_road = [0] * M
    last_road_short = [0] * M
    
    for i in range(M):
        s, t = journeys[i]
        if s < t:
            first_road[i] = s
            if t - s == 2:
                last_road_short[i] = s + 1
        else: # s > t
            first_road[i] = s - 1
            if s - t == 2:
                last_road_short[i] = t
    
    min_F = [M + 1] * (N + 1)
    min_L = [M + 1] * (N + 1)
    
    for i in range(M):
        if first_road[i] != 0:
            k = first_road[i]
            min_F[k] = min(min_F[k], i)
        if last_road_short[i] != 0:
            k = last_road_short[i]
            min_L[k] = min(min_L[k], i)

    conflict_points = []
    for k in range(1, N):
        if min_F[k] <= M and min_L[k] <= M:
            x = min(min_F[k], min_L[k])
            y = max(min_F[k], min_L[k])
            conflict_points.append((x, y))

    queries.sort(key=lambda item: item[1])
    conflict_points.sort(key=lambda item: item[1])
    
    answers = [""] * Q
    point_idx = 0
    
    ft_sum = [0] * (M + 2)
    
    def update_sum(idx, val):
        idx += 1
        while idx <= M + 1:
            ft_sum[idx] += val
            idx += idx & -idx

    def query_sum(idx):
        idx += 1
        res = 0
        while idx > 0:
            res += ft_sum[idx]
            idx -= idx & -idx
        return res
    
    def range_sum(l, r):
        if l > r: return 0
        return query_sum(r) - query_sum(l - 1)

    for l_q, r_q, q_idx in queries:
        while point_idx < len(conflict_points) and conflict_points[point_idx][1] <= r_q:
            x, _ = conflict_points[point_idx]
            update_sum(x, 1)
            point_idx += 1
        
        if range_sum(l_q, M) > 0:
            answers[q_idx] = "No"
        else:
            answers[q_idx] = "Yes"
            
    for ans in answers:
        print(ans)

# It's a good practice to set a higher recursion limit for deep recursion,
# though not strictly necessary for this iterative solution.
sys.setrecursionlimit(10**6)
solve()