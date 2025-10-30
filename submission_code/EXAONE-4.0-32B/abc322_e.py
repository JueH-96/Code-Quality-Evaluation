import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(-1)
        return
    
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))
    
    plans = []
    for _ in range(N):
        c = int(next(it))
        a_list = [int(next(it)) for _ in range(K)]
        plans.append((c, a_list))
    
    init_state = tuple(0 for _ in range(K))
    dp = {init_state: 0}
    
    for cost, gains in plans:
        new_dp = dp.copy()
        for state, total_cost in dp.items():
            new_state_list = []
            for j in range(K):
                new_val = state[j] + gains[j]
                if new_val > P:
                    new_val = P
                new_state_list.append(new_val)
            new_state = tuple(new_state_list)
            new_total = total_cost + cost
            if new_state in new_dp:
                if new_total < new_dp[new_state]:
                    new_dp[new_state] = new_total
            else:
                new_dp[new_state] = new_total
        dp = new_dp
        
    goal_state = tuple(P for _ in range(K))
    if goal_state in dp:
        print(dp[goal_state])
    else:
        print(-1)

if __name__ == '__main__':
    main()