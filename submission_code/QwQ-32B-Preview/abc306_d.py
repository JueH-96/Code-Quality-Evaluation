def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = list(map(int, data[1::2]))
    Y = list(map(int, data[2::2]))
    
    DP_H_prev = 0
    DP_U_prev = -1 << 60  # Negative infinity
    
    for i in range(N):
        DP_H_new = max(DP_H_prev, DP_U_prev)
        DP_U_new = -1 << 60  # Negative infinity
        
        if X[i] == 0:
            # Eat antidotal while healthy: stay healthy
            DP_H_new = max(DP_H_new, DP_H_prev + Y[i])
            # Eat antidotal while upset: become healthy
            DP_H_new = max(DP_H_new, DP_U_prev + Y[i])
            # Skip: already considered
        else:
            # Eat poisonous while healthy: become upset
            DP_U_new = max(DP_U_new, DP_H_prev + Y[i])
            # Skip: remain in previous state, already considered in DP_U_prev
            # Eat poisonous while upset: die, invalid
        
        # Update for upset state considering skips
        if X[i] == 0:
            # Can skip or eat (already handled above)
            DP_U_new = max(DP_U_new, DP_U_prev)
        else:
            # Can only skip if upset, as eating would lead to death
            DP_U_new = max(DP_U_new, DP_U_prev)
        
        DP_H_prev, DP_U_prev = DP_H_new, DP_U_new
    
    ans = max(0, DP_H_prev, DP_U_prev)
    print(ans)

if __name__ == "__main__":
    main()