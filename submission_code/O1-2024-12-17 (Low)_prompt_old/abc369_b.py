def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    N = int(input_data[0])
    presses = [(int(input_data[i*2+1]), input_data[i*2+2]) for i in range(N)]
    
    # dp[l][r] will hold the minimum fatigue after processing a certain number of presses,
    # with the left hand on key l and the right hand on key r.
    # We will update it for each press.
    
    INF = 10**15
    # Initialize dp for the "0th" press: we can place hands anywhere without cost.
    dp = [[0]*101 for _ in range(101)]
    
    # Because the dp is partially updated on each iteration, we must first set all to INF except 0 for all pairs:
    for l in range(1, 101):
        for r in range(1, 101):
            dp[l][r] = 0  # at the start (no presses), 0 cost for any initial placement
    
    for i in range(N):
        key, hand = presses[i]
        new_dp = [[INF]*101 for _ in range(101)]
        
        if hand == 'L':
            # The left hand presses key 'key' => new left hand position is exactly 'key'
            for lpos in range(1, 101):
                for rpos in range(1, 101):
                    cost_before = dp[lpos][rpos]
                    if cost_before == INF:
                        continue
                    move_cost = abs(key - lpos)
                    # Update the new state
                    if move_cost + cost_before < new_dp[key][rpos]:
                        new_dp[key][rpos] = move_cost + cost_before
        else:  # hand == 'R'
            # The right hand presses key 'key' => new right hand position is exactly 'key'
            for lpos in range(1, 101):
                for rpos in range(1, 101):
                    cost_before = dp[lpos][rpos]
                    if cost_before == INF:
                        continue
                    move_cost = abs(key - rpos)
                    # Update the new state
                    if move_cost + cost_before < new_dp[lpos][key]:
                        new_dp[lpos][key] = move_cost + cost_before
        
        dp = new_dp
    
    # The answer is the minimum dp value after all presses
    answer = INF
    for lpos in range(1, 101):
        for rpos in range(1, 101):
            if dp[lpos][rpos] < answer:
                answer = dp[lpos][rpos]
    
    print(answer)

# Let's call solve() here
solve()