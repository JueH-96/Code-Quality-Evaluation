import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(-1)
        return
        
    it = iter(data)
    N = int(next(it)); K = int(next(it)); P = int(next(it))
    base = P + 1
    size = base ** K
    
    goal_index = 0
    for j in range(K):
        goal_index = goal_index * base + P
        
    BIG = 10**18
    dp = [BIG] * size
    dp[0] = 0
    
    plans = []
    for i in range(N):
        c = int(next(it))
        a_list = [int(next(it)) for _ in range(K)]
        plans.append((c, a_list))
        
    for c, a_list in plans:
        new_dp = dp[:]
        for idx in range(size):
            if dp[idx] == BIG:
                continue
            state = [0] * K
            temp = idx
            for j in range(K-1, -1, -1):
                state[j] = temp % base
                temp //= base
                
            new_state = []
            for j in range(K):
                new_val = state[j] + a_list[j]
                if new_val > P:
                    new_val = P
                new_state.append(new_val)
                
            new_index = 0
            for j in range(K):
                new_index = new_index * base + new_state[j]
                
            total_cost = dp[idx] + c
            if total_cost < new_dp[new_index]:
                new_dp[new_index] = total_cost
                
        dp = new_dp
        
    if dp[goal_index] == BIG:
        print(-1)
    else:
        print(dp[goal_index])

if __name__ == "__main__":
    main()