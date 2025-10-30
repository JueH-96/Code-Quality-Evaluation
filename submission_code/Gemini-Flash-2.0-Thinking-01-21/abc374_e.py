import math

def solve():
    n, x = map(int, input().split())
    processes = []
    for _ in range(n):
        processes.append(list(map(int, input().split())))
    
    def get_cost_for_capacity(capacity):
        total_cost = 0
        for i in range(n):
            a_i, p_i, b_i, q_i = processes[i]
            if a_i == 0 and b_i == 0:
                if capacity > 0:
                    return float('inf')
                else:
                    continue
            min_cost_process_i = float('inf')
            if a_i > 0 and b_i > 0:
                if p_i / a_i <= q_i / b_i:
                    for n_s in range(0, math.ceil(capacity / a_i) + 2 if a_i > 0 else 2):
                        n_t = max(0, math.ceil(max(0, capacity - n_s * a_i) / b_i)) if b_i > 0 else 0
                        current_cost = n_s * p_i + n_t * q_i
                        min_cost_process_i = min(min_cost_process_i, current_cost)
                else:
                    for n_t in range(0, math.ceil(capacity / b_i) + 2 if b_i > 0 else 2):
                        n_s = max(0, math.ceil(max(0, capacity - n_t * b_i) / a_i)) if a_i > 0 else 0
                        current_cost = n_s * p_i + n_t * q_i
                        min_cost_process_i = min(min_cost_process_i, current_cost)
            elif a_i > 0:
                min_cost_process_i = p_i * math.ceil(capacity / a_i) if capacity > 0 else 0
            elif b_i > 0:
                min_cost_process_i = q_i * math.ceil(capacity / b_i) if capacity > 0 else 0
            else:
                min_cost_process_i = 0
                
            total_cost += min_cost_process_i
        return total_cost

    low = 0
    high = 2 * 10**9
    max_capacity = 0
    
    while low <= high:
        mid_capacity = (low + high) // 2
        cost = get_cost_for_capacity(mid_capacity)
        if cost <= x:
            max_capacity = mid_capacity
            low = mid_capacity + 1
        else:
            high = mid_capacity - 1
            
    print(max_capacity)

if __name__ == '__main__':
    solve()