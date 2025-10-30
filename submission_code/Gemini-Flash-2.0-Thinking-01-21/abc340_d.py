import heapq

def solve():
    n = int(input())
    if n == 1:
        print(0)
        return
    a_costs = []
    b_costs = []
    x_stages = []
    for _ in range(n - 1):
        a, b, x = map(int, input().split())
        a_costs.append(a)
        b_costs.append(b)
        x_stages.append(x)
    
    min_times = [float('inf')] * (n + 1)
    min_times[1] = 0
    pq = [(0, 1)] # (time, stage)
    visited = [False] * (n + 1)
    
    while pq:
        current_time, current_stage = heapq.heappop(pq)
        if current_stage == n:
            print(current_time)
            return
        if visited[current_stage]:
            continue
        visited[current_stage] = True
        
        if current_stage < n:
            # Option 1: to stage current_stage + 1
            next_stage_1 = current_stage + 1
            cost_1 = a_costs[current_stage - 1]
            new_time_1 = current_time + cost_1
            if new_time_1 < min_times[next_stage_1]:
                min_times[next_stage_1] = new_time_1
                heapq.heappush(pq, (new_time_1, next_stage_1))
                
            # Option 2: to stage x_stages[current_stage - 1]
            next_stage_2 = x_stages[current_stage - 1]
            cost_2 = b_costs[current_stage - 1]
            new_time_2 = current_time + cost_2
            if new_time_2 < min_times[next_stage_2]:
                min_times[next_stage_2] = new_time_2
                heapq.heappush(pq, (new_time_2, next_stage_2))
                
    return -1 # Should not reach here as stage N is always reachable

if __name__ == '__main__':
    solve()