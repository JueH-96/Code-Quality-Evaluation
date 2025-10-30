def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Read team assignments A and strengths B
    A = []
    B = []
    idx = 1
    for _ in range(N):
        Ai = int(input_data[idx]); Bi = int(input_data[idx+1])
        A.append(Ai - 1)  # convert to 0-based teams: 0,1,2
        B.append(Bi)
        idx += 2
    
    total_strength = sum(B)
    # If total is not divisible by 3, it's impossible
    if total_strength % 3 != 0:
        print(-1)
        return
    
    T = total_strength // 3
    # Prefix sums: sum_of[i] = B[0] + ... + B[i-1]
    # So sum_of[i+1] - sum_of[i] = B[i]
    sum_of = [0] * (N+1)
    for i in range(N):
        sum_of[i+1] = sum_of[i] + B[i]
    
    # We'll do a DP approach where dp[x][y] = minimum cost
    # to distribute first i items so team1 has sum x, team2 has sum y,
    # team3 has sum_of[i] - x - y.
    # We only keep dp for the "current" i, rolling update.
    INF = 10**9
    # Flatten 2D array of size (T+1)*(T+1)
    # index in 1D = x*(T+1)+y
    tsize = T+1
    
    dp_curr = [INF]*(tsize*tsize)
    dp_next = [INF]*(tsize*tsize)
    
    # Initially, with 0 items, cost 0 if x=0,y=0
    dp_curr[0] = 0
    
    for i in range(N):
        # Clear dp_next
        for k in range(tsize*tsize):
            dp_next[k] = INF
        Bi = B[i]
        Ai = A[i]
        sum_i_plus_1 = sum_of[i+1]  # sum of first i+1 items
        
        for x in range(T+1):
            # Because x + y <= sum_i_plus_1 always; no need to check beyond that
            base_idx = x*(T+1)
            for y in range(T+1):
                base_cost = dp_curr[base_idx + y]
                if base_cost == INF:
                    continue
                # We'll place item i in team0, team1, or team2
                # Check each possible next (x', y')
                
                # team 0 => new sums: x' = x + Bi, y' = y
                x0 = x + Bi
                y0 = y
                if x0 <= T:
                    # sum3 = sum_i_plus_1 - x0 - y0
                    if sum_i_plus_1 - x0 - y0 <= T:
                        new_cost = base_cost + (0 if Ai == 0 else 1)
                        idx0 = x0*(T+1) + y0
                        if dp_next[idx0] > new_cost:
                            dp_next[idx0] = new_cost
                
                # team 1 => x' = x, y' = y + Bi
                x1 = x
                y1 = y + Bi
                if y1 <= T:
                    if sum_i_plus_1 - x1 - y1 <= T:
                        new_cost = base_cost + (0 if Ai == 1 else 1)
                        idx1 = x1*(T+1) + y1
                        if dp_next[idx1] > new_cost:
                            dp_next[idx1] = new_cost
                
                # team 2 => x' = x, y' = y
                # sum3' will be sum_i_plus_1 - x - y, which includes Bi in team3
                if sum_i_plus_1 - x - y <= T:
                    new_cost = base_cost + (0 if Ai == 2 else 1)
                    idx2 = x*(T+1) + y
                    if dp_next[idx2] > new_cost:
                        dp_next[idx2] = new_cost
        
        # Swap dp_next into dp_curr
        dp_curr, dp_next = dp_next, dp_curr
    
    # At the end, we want dp_curr[T][T]
    ans = dp_curr[T*(T+1) + T]
    if ans >= INF:
        print(-1)
    else:
        print(ans)

# Do not forget to call main()
main()