def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    P = int(input[idx]); idx +=1

    plans = []
    for _ in range(N):
        C = int(input[idx]); idx +=1
        A = list(map(int, input[idx:idx+K]))
        idx += K
        plans.append( (C, A) )

    # Initialize DP
    dp = {tuple([0]*K): 0}

    for C_i, A_i in plans:
        temp_dp = dp.copy()
        for state in dp:
            current = list(state)
            new_state = list(current)
            for j in range(K):
                new_state[j] = current[j] + A_i[j]
                if new_state[j] > P:
                    new_state[j] = P
            new_state = tuple(new_state)
            new_cost = dp[state] + C_i
            if new_state in temp_dp:
                if new_cost < temp_dp[new_state]:
                    temp_dp[new_state] = new_cost
            else:
                temp_dp[new_state] = new_cost
        dp = temp_dp

    target = tuple([P]*K)
    if target in dp:
        print(dp[target])
    else:
        print(-1)

if __name__ == '__main__':
    main()