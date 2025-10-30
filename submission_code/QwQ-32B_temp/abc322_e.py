import sys

def main():
    N, K, P = map(int, sys.stdin.readline().split())
    plans = []
    for _ in range(N):
        parts = list(map(int, sys.stdin.readline().split()))
        C = parts[0]
        A = parts[1:1+K]
        plans.append((C, A))
    
    # Initialize DP: state is a tuple of K elements, each 0..P
    initial = tuple([0] * K)
    dp = {initial: 0}
    
    for (c, a) in plans:
        new_dp = dp.copy()  # Start with existing states (not taking the current plan)
        for state in dp:
            current_cost = dp[state]
            new_state_list = list(state)
            for j in range(K):
                new_val = new_state_list[j] + a[j]
                if new_val > P:
                    new_val = P
                new_state_list[j] = new_val
            new_state = tuple(new_state_list)
            new_cost = current_cost + c
            # Update new_dp if this path is better
            if new_state in new_dp:
                if new_cost < new_dp[new_state]:
                    new_dp[new_state] = new_cost
            else:
                new_dp[new_state] = new_cost
        dp = new_dp
    
    target = tuple([P] * K)
    if target in dp:
        print(dp[target])
    else:
        print(-1)

if __name__ == "__main__":
    main()