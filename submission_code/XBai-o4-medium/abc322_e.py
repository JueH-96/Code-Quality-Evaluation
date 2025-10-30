import sys

def main():
    n, k, p = map(int, sys.stdin.readline().split())
    plans = []
    for _ in range(n):
        parts = list(map(int, sys.stdin.readline().split()))
        c = parts[0]
        a = parts[1:1+k]
        plans.append((c, a))
    
    # Initialize DP dictionary
    dp = {}
    initial_state = tuple([0]*k)
    dp[initial_state] = 0
    
    for cost_i, a_i in plans:
        current_states = list(dp.items())
        for state, cur_cost in current_states:
            new_state_list = list(state)
            for j in range(k):
                new_state_list[j] = min(new_state_list[j] + a_i[j], p)
            new_state = tuple(new_state_list)
            new_cost = cur_cost + cost_i
            if new_state in dp:
                if new_cost < dp[new_state]:
                    dp[new_state] = new_cost
            else:
                dp[new_state] = new_cost
    
    target = tuple([p]*k)
    print(dp.get(target, -1))

if __name__ == "__main__":
    main()