def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    K = int(input[idx+1])
    P = int(input[idx+2])
    idx +=3
    
    plans = []
    for _ in range(N):
        C = int(input[idx])
        A = list(map(int, input[idx+1:idx+1+K]))
        plans.append( (C, A) )
        idx += 1+K
    
    from collections import defaultdict
    
    initial_state = tuple([0]*K)
    dp = { initial_state: 0 }
    
    for (cost, a_list) in plans:
        tmp_dp = dict(dp)  # copy existing states
        # iterate through the previous states in dp
        for state in dp:
            current_cost = dp[state]
            # compute new_state
            new_state = list(state)
            for j in range(K):
                new_state[j] = min( new_state[j] + a_list[j], P )
            new_state = tuple(new_state)
            new_total_cost = current_cost + cost
            # update tmp_dp
            if new_state in tmp_dp:
                if new_total_cost < tmp_dp[new_state]:
                    tmp_dp[new_state] = new_total_cost
            else:
                tmp_dp[new_state] = new_total_cost
        # replace dp with tmp_dp
        dp = tmp_dp
    
    min_cost = float('inf')
    target = tuple([P]*K)
    found = False
    for state in dp:
        if state == target:
            if dp[state] < min_cost:
                min_cost = dp[state]
                found = True
    
    if found:
        print(min_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()