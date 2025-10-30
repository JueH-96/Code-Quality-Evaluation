import sys

def solve():
    n = int(sys.stdin.readline())
    presses = []
    for _ in range(n):
        line = sys.stdin.readline().split()
        key = int(line[0])
        hand = line[1]
        presses.append({'key': key, 'hand': hand})
    
    dp = {}
    
    # Initialize dp for step 0 (before any press). Initial fatigue is 0 for any starting hand positions.
    for l_pos in range(1, 101):
        for r_pos in range(1, 101):
            dp[(0, l_pos, r_pos)] = 0
            
    for i in range(1, n + 1):
        current_press = presses[i-1]
        key_to_press = current_press['key']
        hand_to_use = current_press['hand']
        
        for prev_l_pos in range(1, 101):
            for prev_r_pos in range(1, 101):
                if (i-1, prev_l_pos, prev_r_pos) in dp:
                    current_fatigue = dp[(i-1, prev_l_pos, prev_r_pos)]
                    
                    if hand_to_use == 'L':
                        fatigue_increase_l = abs(key_to_press - prev_l_pos)
                        for next_r_pos in range(1, 101):
                            fatigue_increase_r = abs(next_r_pos - prev_r_pos)
                            next_fatigue = current_fatigue + fatigue_increase_l + fatigue_increase_r
                            state = (i, key_to_press, next_r_pos)
                            if state not in dp or next_fatigue < dp[state]:
                                dp[state] = next_fatigue
                    elif hand_to_use == 'R':
                        fatigue_increase_r = abs(key_to_press - prev_r_pos)
                        for next_l_pos in range(1, 101):
                            fatigue_increase_l = abs(next_l_pos - prev_l_pos)
                            next_fatigue = current_fatigue + fatigue_increase_r + fatigue_increase_l
                            state = (i, next_l_pos, key_to_press)
                            if state not in dp or next_fatigue < dp[state]:
                                dp[state] = next_fatigue
                                
    min_fatigue = float('inf')
    for l_pos in range(1, 101):
        for r_pos in range(1, 101):
            if (n, l_pos, r_pos) in dp:
                min_fatigue = min(min_fatigue, dp[(n, l_pos, r_pos)])
                
    print(min_fatigue)

if __name__ == '__main__':
    solve()