import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    
    instructions = []
    for _ in range(Q):
        H = data[idx]
        T = int(data[idx+1])
        instructions.append((H, T))
        idx += 2
    
    # Precompute distance matrix
    def distance(u, v):
        d_clock = (v - u) % N
        d_counter = (u - v) % N
        return min(d_clock, d_counter)
    
    dist = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(1, N+1):
            dist[i][j] = distance(i, j)
    
    INF = float('inf')
    
    # Initialize previous DP
    # Handle the first instruction separately
    H0, T0 = instructions[0]
    prev_dp = [INF] * (N + 1)
    if H0 == 'L':
        # previous other hand (R) is at 2
        prev_dp[2] = 0
    else:
        # previous other hand (L) is at 1
        prev_dp[1] = 0
    
    for step in range(Q):
        H, T = instructions[step]
        current_dp = [INF] * (N + 1)
        
        if step == 0:
            # Handle first instruction
            for s_prev in range(1, N+1):
                if prev_dp[s_prev] == INF:
                    continue
                # Determine previous moved position
                if H == 'L':
                    # Moving L from initial position 1
                    prev_moved = 1
                else:
                    # Moving R from initial position 2
                    prev_moved = 2
                dH = dist[prev_moved][T]
                # Iterate all s_new for other hand (not T)
                for s_new in range(1, N+1):
                    if s_new == T:
                        continue
                    dO = dist[s_prev][s_new]
                    total = prev_dp[s_prev] + dH + dO
                    if total < current_dp[s_new]:
                        current_dp[s_new] = total
        else:
            # Handle subsequent instructions
            prev_H, prev_T = instructions[step-1]
            for s_prev in range(1, N+1):
                if prev_dp[s_prev] == INF:
                    continue
                # Determine prev_moved position
                if H == prev_H:
                    prev_moved = prev_T
                else:
                    prev_moved = s_prev
                dH = dist[prev_moved][T]
                # Iterate all s_new for other hand (not T)
                for s_new in range(1, N+1):
                    if s_new == T:
                        continue
                    dO = dist[s_prev][s_new]
                    total = prev_dp[s_prev] + dH + dO
                    if total < current_dp[s_new]:
                        current_dp[s_new] = total
        
        # Update prev_dp for next iteration
        prev_dp = current_dp
    
    # Find the minimum in prev_dp (excluding INF)
    min_steps = INF
    for val in prev_dp[1:]:
        if val < min_steps:
            min_steps = val
    print(min_steps)

if __name__ == '__main__':
    main()