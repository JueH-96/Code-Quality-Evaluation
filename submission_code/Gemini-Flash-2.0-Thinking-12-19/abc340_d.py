import heapq

def solve():
    n = int(input())
    if n == 1:
        print(0)
        return
    a_values = []
    b_values = []
    x_values = []
    for _ in range(n - 1):
        a, b, x = map(int, input().split())
        a_values.append(a)
        b_values.append(b)
        x_values.append(x)
    
    min_time = [float('inf')] * (n + 1)
    min_time[1] = 0
    pq = [(0, 1)]
    
    while pq:
        current_time, current_stage = heapq.heappop(pq)
        if current_time > min_time[current_stage]:
            continue
        if current_stage == n:
            print(current_time)
            return
            
        if current_stage < n:
            # Action 1: to stage current_stage + 1
            next_stage_1 = current_stage + 1
            cost_1 = a_values[current_stage - 1]
            if next_stage_1 <= n:
                if current_time + cost_1 < min_time[next_stage_1]:
                    min_time[next_stage_1] = current_time + cost_1
                    heapq.heappush(pq, (min_time[next_stage_1], next_stage_1))
                    
            # Action 2: to stage x_values[current_stage - 1]
            next_stage_2 = x_values[current_stage - 1]
            cost_2 = b_values[current_stage - 1]
            if next_stage_2 <= n:
                if current_time + cost_2 < min_time[next_stage_2]:
                    min_time[next_stage_2] = current_time + cost_2
                    heapq.heappush(pq, (min_time[next_stage_2], next_stage_2))
                    
    return -1 # Should not reach here in this problem

if __name__ == '__main__':
    solve()