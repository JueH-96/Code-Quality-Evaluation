from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    P = int(input[idx]); idx +=1
    
    plans = []
    for _ in range(N):
        C_i = int(input[idx]); idx +=1
        A_i = list(map(int, input[idx:idx+K]))
        idx += K
        plans.append( (C_i, A_i) )
    
    dp = defaultdict(lambda: float('inf'))
    initial_state = tuple([0]*K)
    dp[initial_state] = 0
    
    for C_i, A_i in plans:
        temp = {}
        for state in list(dp.keys()):
            current_cost = dp[state]
            new_state = list(state)
            for j in range(K):
                new_state[j] += A_i[j]
                if new_state[j] > P:
                    new_state[j] = P
            new_state = tuple(new_state)
            new_cost = current_cost + C_i
            if new_cost < temp.get(new_state, float('inf')):
                temp[new_state] = new_cost
        for s in temp:
            if s not in dp or temp[s] < dp[s]:
                dp[s] = temp[s]
    
    target = tuple([P]*K)
    if target in dp:
        print(dp[target])
    else:
        print(-1)

if __name__ == '__main__':
    main()