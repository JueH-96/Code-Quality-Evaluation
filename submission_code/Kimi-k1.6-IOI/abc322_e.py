import sys

def main():
    n, k, p = map(int, sys.stdin.readline().split())
    plans = []
    for _ in range(n):
        parts = list(map(int, sys.stdin.readline().split()))
        cost = parts[0]
        a = parts[1:1+k]
        plans.append((cost, a))
    
    # Initialize DP: key is tuple of parameters, value is cost
    dp = {tuple([0]*k): 0}
    
    for cost, a in plans:
        temp_dp = dp.copy()
        for state in list(dp.keys()):
            new_state = list(state)
            for j in range(k):
                new_state[j] = min(new_state[j] + a[j], p)
            new_state = tuple(new_state)
            new_cost = dp[state] + cost
            if new_state in temp_dp:
                if new_cost < temp_dp[new_state]:
                    temp_dp[new_state] = new_cost
            else:
                temp_dp[new_state] = new_cost
        dp = temp_dp
    
    min_cost = None
    for state in dp:
        valid = True
        for val in state:
            if val < p:
                valid = False
                break
        if valid:
            if min_cost is None or dp[state] < min_cost:
                min_cost = dp[state]
    
    print(min_cost if min_cost is not None else -1)

if __name__ == "__main__":
    main()