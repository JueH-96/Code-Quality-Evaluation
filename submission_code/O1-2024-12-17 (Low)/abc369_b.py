def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    presses = [(int(input_data[2*i+1]), input_data[2*i+2]) for i in range(N)]
    
    # dp[i][l][r] = minimum fatigue after i-th press (0-based) 
    # when left hand is on position l and right hand is on position r.
    # We'll use 0 to represent "unplaced" (which does not incur a movement cost for the first placement).
    INF = 10**15
    dp = [[[INF]*101 for _ in range(101)] for _ in range(N+1)]
    
    # Initially, no keys have been pressed, and both hands are "unplaced".
    dp[0][0][0] = 0
    
    for i in range(N):
        key, hand = presses[i]
        for lpos in range(101):
            for rpos in range(101):
                if dp[i][lpos][rpos] == INF:
                    continue
                cost_so_far = dp[i][lpos][rpos]
                
                if hand == 'L':
                    # Move left hand from lpos to key
                    move_cost = 0 if lpos == 0 else abs(key - lpos)
                    new_lpos = key
                    dp[i+1][new_lpos][rpos] = min(dp[i+1][new_lpos][rpos], cost_so_far + move_cost)
                else:
                    # Move right hand from rpos to key
                    move_cost = 0 if rpos == 0 else abs(key - rpos)
                    new_rpos = key
                    dp[i+1][lpos][new_rpos] = min(dp[i+1][lpos][new_rpos], cost_so_far + move_cost)
    
    # The answer is the minimum over all possible positions of the hands after the N-th press
    ans = INF
    for lpos in range(101):
        for rpos in range(101):
            ans = min(ans, dp[N][lpos][rpos])
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()