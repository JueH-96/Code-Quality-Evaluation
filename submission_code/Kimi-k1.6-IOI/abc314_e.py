import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    wheels = []
    for _ in range(N):
        C_i = int(input[ptr])
        ptr += 1
        P_i = int(input[ptr])
        ptr += 1
        S_i = list(map(int, input[ptr:ptr+P_i]))
        ptr += P_i
        wheels.append((C_i, P_i, S_i))
    
    INF = float('inf')
    dp_prev = [INF] * (M + 1)
    dp_prev[M] = 0.0  # Base case: already have M points
    
    epsilon = 1e-10
    converged = False
    max_iterations = 100000  # Arbitrary large number to ensure convergence
    
    for _ in range(max_iterations):
        dp_next = [0.0] * (M + 1)
        dp_next[M] = 0.0  # Always zero
        
        for k in range(M):
            min_cost = INF
            for (C_i, P_i, S_i) in wheels:
                total = 0.0
                for s in S_i:
                    next_state = k + s
                    if next_state >= M:
                        next_state = M
                    total += dp_prev[next_state]
                expected_cost = C_i + (total / P_i)
                if expected_cost < min_cost:
                    min_cost = expected_cost
            dp_next[k] = min_cost
        
        # Check for convergence
        max_diff = 0.0
        for k in range(M + 1):
            diff = abs(dp_next[k] - dp_prev[k])
            if diff > max_diff:
                max_diff = diff
        if max_diff < epsilon:
            break
        
        # Update previous DP array
        dp_prev = dp_next[:]
    
    # Print with sufficient precision
    print("{0:.10f}".format(dp_prev[0]))

if __name__ == "__main__":
    main()