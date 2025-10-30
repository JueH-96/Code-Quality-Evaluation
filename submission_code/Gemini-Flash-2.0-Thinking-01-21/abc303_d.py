import math

def solve():
    x, y, z = map(int, input().split())
    s = input()
    n = len(s)
    dp = {}
    dp[(0, False)] = 0
    dp[(0, True)] = math.inf
    
    for i in range(n):
        next_dp = {}
        for caps_on in [False, True]:
            if (i, caps_on) not in dp:
                continue
            current_cost = dp[(i, caps_on)]
            if current_cost == math.inf:
                continue
                
            # Operation 1: Press 'a' key (cost X)
            char_typed_op1 = 'A' if caps_on else 'a'
            next_caps_on_op1 = caps_on
            if char_typed_op1 == s[i]:
                next_state = (i + 1, next_caps_on_op1)
                next_cost = current_cost + x
                next_dp[next_state] = min(next_dp.get(next_state, math.inf), next_cost)
                
            # Operation 2: Press 'a' and Shift key (cost Y)
            char_typed_op2 = 'a' if caps_on else 'A'
            next_caps_on_op2 = caps_on
            if char_typed_op2 == s[i]:
                next_state = (i + 1, next_caps_on_op2)
                next_cost = current_cost + y
                next_dp[next_state] = min(next_dp.get(next_state, math.inf), next_cost)
                
            # Operation 3: Press Caps Lock key (cost Z)
            next_caps_on_op3 = not caps_on
            next_state_op3 = (i, next_caps_on_op3)
            next_cost_op3 = current_cost + z
            next_dp[next_state_op3] = min(next_dp.get(next_state_op3, math.inf), next_cost_op3)
            
        for state, cost in next_dp.items():
            dp[state] = min(dp.get(state, math.inf), cost)
            
    result = min(dp.get((n, False), math.inf), dp.get((n, True), math.inf))
    print(result)

if __name__ == '__main__':
    solve()