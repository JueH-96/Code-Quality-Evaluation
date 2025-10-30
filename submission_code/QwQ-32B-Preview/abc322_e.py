import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = int(data[2])
    
    plans = []
    idx = 3
    for _ in range(N):
        C = int(data[idx])
        A = list(map(int, data[idx+1:idx+1+K]))
        plans.append((C, A))
        idx += 1 + K
    
    base = P + 1
    num_states = base ** K
    goal = base ** K - 1  # Since goal is when all params are P
    
    dp = [math.inf] * num_states
    dp[0] = 0  # Initial state with all params at 0
    
    for C, A in plans:
        for s in range(num_states-1, -1, -1):
            if dp[s] == math.inf:
                continue
            # Compute new state s'
            new_state = 0
            multiplier = 1
            current_state = s
            for j in range(K):
                param_j = current_state % base
                new_param_j = min(param_j + A[j], P)
                new_state += new_param_j * multiplier
                multiplier *= base
                current_state //= base
            # Update dp[new_state]
            dp[new_state] = min(dp[new_state], dp[s] + C)
    
    if dp[goal] != math.inf:
        print(dp[goal])
    else:
        print(-1)

if __name__ == "__main__":
    main()