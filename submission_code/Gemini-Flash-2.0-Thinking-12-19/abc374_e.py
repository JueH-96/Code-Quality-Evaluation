import math

def solve():
    n, x = map(int, input().split())
    processes = []
    for _ in range(n):
        processes.append(list(map(int, input().split())))
    
    def get_min_cost(process_index, capacity):
        a_i, p_i, b_i, q_i = processes[process_index]
        if capacity <= 0:
            return 0
        if a_i == 0 and b_i == 0:
            return float('inf')
        min_cost = float('inf')
        max_s_i = 0
        if a_i > 0:
            max_s_i = math.ceil(capacity / a_i)
        else:
            max_s_i = 0
            
        for s_i in range(max_s_i + 1):
            t_i = 0
            if b_i > 0:
                required_capacity_t = max(0, capacity - s_i * a_i)
                t_i = max(0, math.ceil(required_capacity_t / b_i))
            current_cost = s_i * p_i + t_i * q_i
            min_cost = min(min_cost, current_cost)
            
        if min_cost == float('inf'):
            if a_i > 0:
                cost_only_s = p_i * math.ceil(capacity / a_i)
                min_cost = min(min_cost, cost_only_s)
            if b_i > 0:
                cost_only_t = q_i * math.ceil(capacity / b_i)
                min_cost = min(min_cost, cost_only_t)
                
        if min_cost == float('inf'):
            return float('inf')
        return min_cost

    max_capacity = 0
    low = 0
    high = 2 * 10**9 # A large upper bound for capacity
    
    while low <= high:
        mid_capacity = (low + high) // 2
        total_cost = 0
        possible = True
        for i in range(n):
            cost_i = get_min_cost(i, mid_capacity)
            if cost_i == float('inf'):
                possible = False
                break
            total_cost += cost_i
        
        if possible and total_cost <= x:
            max_capacity = mid_capacity
            low = mid_capacity + 1
        else:
            high = mid_capacity - 1
            
    print(max_capacity)

if __name__ == '__main__':
    solve()