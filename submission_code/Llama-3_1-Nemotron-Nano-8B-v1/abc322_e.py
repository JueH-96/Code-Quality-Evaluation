import sys
from collections import defaultdict

def main():
    n, k, p = map(int, sys.stdin.readline().split())
    plans = []
    for _ in range(n):
        parts = list(map(int, sys.stdin.readline().split()))
        c = parts[0]
        a = parts[1:1+k]
        plans.append((c, a))
    
    dp = defaultdict(lambda: float('inf'))
    initial_state = tuple([0] * k)
    dp[initial_state] = 0
    
    for c, a in plans:
        new_states = {}
        for state in list(dp.keys()):
            current_cost = dp[state]
            new_state = list(state)
            for j in range(k):
                new_val = min(new_state[j] + a[j], p)
                new_state[j] = new_val
            new_state = tuple(new_state)
            new_cost = current_cost + c
            if new_state in new_states:
                if new_cost < new_states[new_state]:
                    new_states[new_state] = new_cost
            else:
                new_states[new_state] = new_cost
        
        for s in new_states:
            if s in dp:
                if new_states[s] < dp[s]:
                    dp[s] = new_states[s]
            else:
                dp[s] = new_states[s]
    
    target = tuple([p] * k)
    if target in dp:
        print(dp[target])
    else:
        print(-1)

if __name__ == "__main__":
    main()