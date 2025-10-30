import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(n):
        line = list(map(int, sys.stdin.readline().split()))
        c_i = line[0]
        p_i = line[1]
        s_i = line[2:]
        wheels.append({'cost': c_i, 'probabilities': 1.0/p_i, 'scores': s_i})
    
    dp = [0.0] * (m + 1)
    
    for p in range(m - 1, -1, -1):
        min_expected_cost = float('inf')
        for i in range(n):
            wheel = wheels[i]
            expected_cost_wheel_i = wheel['cost']
            sum_expected_future_cost = 0.0
            for score in wheel['scores']:
                next_points = min(m, p + score)
                sum_expected_future_cost += dp[next_points]
            expected_cost_wheel_i += wheel['probabilities'] * sum_expected_future_cost
            min_expected_cost = min(min_expected_cost, expected_cost_wheel_i)
        dp[p] = min_expected_cost
        
    print(f"{dp[0]:.12f}")

if __name__ == '__main__':
    solve()