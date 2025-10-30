def solve():
    n, k, p = map(int, input().split())
    plans = []
    for _ in range(n):
        line = list(map(int, input().split()))
        plans.append((line[0], line[1:]))

    min_cost = float('inf')

    for i in range(1 << n):
        current_cost = 0
        current_params = [0] * k
        
        selected_plans = []
        for j in range(n):
            if (i >> j) & 1:
                selected_plans.append(plans[j])
        
        if not selected_plans:
            continue

        for cost, params in selected_plans:
            current_cost += cost
            for idx in range(k):
                current_params[idx] += params[idx]
        
        
        all_params_met = True
        for param_val in current_params:
            if param_val < p:
                all_params_met = False
                break
        
        if all_params_met:
            min_cost = min(min_cost, current_cost)
    
    if min_cost == float('inf'):
        print("-1")
    else:
        print(min_cost)

solve()