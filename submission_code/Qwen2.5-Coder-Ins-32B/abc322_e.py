# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    import sys
    input = sys.stdin.read().split()
    
    N = int(input[0])
    K = int(input[1])
    P = int(input[2])
    
    plans = []
    idx = 3
    for _ in range(N):
        C = int(input[idx])
        A = list(map(int, input[idx+1:idx+1+K]))
        plans.append((C, A))
        idx += 1 + K
    
    min_cost = float('inf')
    found = False
    
    for r in range(1, N + 1):
        for combo in combinations(plans, r):
            total_cost = sum(cost for cost, _ in combo)
            total_params = [0] * K
            for _, params in combo:
                for j in range(K):
                    total_params[j] += params[j]
            if all(param >= P for param in total_params):
                found = True
                min_cost = min(min_cost, total_cost)
    
    if found:
        print(min_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()