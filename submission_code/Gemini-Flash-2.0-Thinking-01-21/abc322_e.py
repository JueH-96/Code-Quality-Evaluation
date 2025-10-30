import math

def solve():
    n, k, p = map(int, input().split())
    plans = []
    for _ in range(n):
        line = list(map(int, input().split()))
        cost = line[0]
        increments = line[1:]
        plans.append({'cost': cost, 'increments': increments})
    
    dp = {}
    initial_state = tuple([0] * k)
    dp[initial_state] = 0
    
    for i in range(n):
        next_dp = dp.copy()
        current_plan = plans[i]
        cost_i = current_plan['cost']
        increments_i = current_plan['increments']
        
        for state in list(dp.keys()):
            current_cost = dp[state]
            
            # Option 1: Don't use plan i. Cost remains current_cost, state remains state. (Already considered in next_dp initialization)
            
            # Option 2: Use plan i. 
            next_state_list = []
            for j in range(k):
                next_val = min(p, state[j] + increments_i[j])
                next_state_list.append(next_val)
            next_state = tuple(next_state_list)
            new_cost = current_cost + cost_i
            
            if next_state not in next_dp or new_cost < next_dp[next_state]:
                next_dp[next_state] = new_cost
                
        dp = next_dp
        
    min_cost = float('inf')
    target_state_prefix = tuple([p] * k)
    
    for state, cost in dp.items():
        is_goal_reached = True
        for j in range(k):
            if state[j] < p:
                is_goal_reached = False
                break
        if is_goal_reached:
            min_cost = min(min_cost, cost)
            
    if min_cost == float('inf'):
        print("-1")
    else:
        print(min_cost)

if __name__ == '__main__':
    solve()