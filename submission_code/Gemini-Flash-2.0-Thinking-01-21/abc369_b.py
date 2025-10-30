import math

def solve():
    n = int(input())
    presses = []
    for _ in range(n):
        line = input().split()
        key = int(line[0])
        hand = line[1]
        presses.append({'key': key, 'hand': hand})
    
    dp = {}
    
    # Initialize dp[0][l][r] = 0 for all possible initial positions l, r
    for l in range(1, 101):
        for r in range(1, 101):
            dp[(0, l, r)] = 0
            
    for i in range(1, n + 1):
        current_press = presses[i-1]
        key_to_press = current_press['key']
        hand_to_use = current_press['hand']
        
        for prev_left_pos in range(1, 101):
            for prev_right_pos in range(1, 101):
                if (i-1, prev_left_pos, prev_right_pos) in dp:
                    current_fatigue = dp[(i-1, prev_left_pos, prev_right_pos)]
                    if hand_to_use == 'L':
                        fatigue_increase = abs(key_to_press - prev_left_pos)
                        next_left_pos = key_to_press
                        next_right_pos = prev_right_pos
                    else: # hand_to_use == 'R'
                        fatigue_increase = abs(key_to_press - prev_right_pos)
                        next_left_pos = prev_left_pos
                        next_right_pos = key_to_press
                        
                    new_fatigue = current_fatigue + fatigue_increase
                    state = (i, next_left_pos, next_right_pos)
                    if state not in dp or new_fatigue < dp[state]:
                        dp[state] = new_fatigue
                        
    min_fatigue = float('inf')
    for l in range(1, 101):
        for r in range(1, 101):
            if (n, l, r) in dp:
                min_fatigue = min(min_fatigue, dp[(n, l, r)])
                
    print(min_fatigue)

if __name__ == '__main__':
    solve()