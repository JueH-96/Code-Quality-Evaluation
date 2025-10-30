def solve():
    n = int(input())
    conditions = []
    for _ in range(n):
        a_i, b_i, c_i = map(int, input().split())
        conditions.append({'a': a_i, 'b': b_i, 'c': c_i})
    
    x_max_values = []
    for cond in conditions:
        x_max_val = (cond['c'] - cond['b'] - 1) // cond['a']
        if x_max_val < 0:
            print(0)
            return
        x_max_values.append(x_max_val)
    
    x_upper_bound = min(x_max_values)
    if x_upper_bound < 1:
        print(0)
        return
        
    total_count = 0
    x_current = 1
    while x_current <= x_upper_bound:
        min_y_max = float('inf')
        for cond in conditions:
            y_max_val = (cond['c'] - cond['a'] * x_current - 1) // cond['b']
            min_y_max = min(min_y_max, y_max_val)
            
        current_y_max = max(0, min_y_max)
        if current_y_max == 0:
            break
            
        u_limit = float('inf')
        for cond in conditions:
            u_i = (cond['c'] - 1 - current_y_max * cond['b']) // cond['a']
            u_limit = min(u_limit, u_i)
            
        range_length = max(0, u_limit - x_current + 1)
        total_count += range_length * current_y_max
        x_current = u_limit + 1
        
    print(total_count)

t = int(input())
for _ in range(t):
    solve()