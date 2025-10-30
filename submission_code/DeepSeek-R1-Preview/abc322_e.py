def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    P = int(input[idx])
    idx += 1
    
    plans = []
    for _ in range(N):
        C_i = int(input[idx])
        idx += 1
        A_i = list(map(int, input[idx:idx+K]))
        idx += K
        plans.append((C_i, A_i))
    
    dp = {tuple([0]*K): 0}
    for C_i, A_i in plans:
        temp = {}
        for state in dp:
            current_cost = dp[state]
            new_state = []
            for s, a in zip(state, A_i):
                new_val = s + a
                if new_val > P:
                    new_val = P
                new_state.append(new_val)
            new_state = tuple(new_state)
            new_cost = current_cost + C_i
            if new_state not in temp or new_cost < temp[new_state]:
                temp[new_state] = new_cost
        for s in temp:
            if s not in dp or temp[s] < dp.get(s, float('inf')):
                dp[s] = temp[s]
    
    min_cost = None
    for state in dp:
        if all(x >= P for x in state):
            if min_cost is None or dp[state] < min_cost:
                min_cost = dp[state]
    
    if min_cost is not None:
        print(min_cost)
    else:
        print(-1)

if __name__ == '__main__':
    main()