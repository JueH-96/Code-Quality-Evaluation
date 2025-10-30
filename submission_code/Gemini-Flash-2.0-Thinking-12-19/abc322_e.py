import math

def solve():
    n, k, p = map(int, input().split())
    plans = []
    for _ in range(n):
        line = list(map(int, input().split()))
        plans.append({'cost': line[0], 'increases': line[1:]})
    
    dp = {}
    initial_state = tuple([0] * k)
    dp[initial_state] = 0
    
    for i in range(n):
        next_dp = dp.copy()
        plan = plans[i]
        cost = plan['cost']
        increases = plan['increases']
        
        for state in dp:
            current_cost = dp[state]
            next_state_list = []
            for j in range(k):
                next_val = min(p, state[j] + increases[j])
                next_state_list.append(next_val)
            next_state = tuple(next_state_list)
            
            new_cost = current_cost + cost
            if next_state not in next_dp:
                next_dp[next_state] = new_cost
            else:
                next_dp[next_state] = min(next_dp[next_state], new_cost)
                
        dp = next_dp
        
    min_cost = float('inf')
    for state in dp:
        all_satisfied = True
        for val in state:
            if val < p:
                all_satisfied = False
                break
        if all_satisfied:
            min_cost = min(min_cost, dp[state])
            
    if min_cost == float('inf'):
        print("-1")
    else:
        print(min_cost)

if __name__ == '__main__':
    solve()