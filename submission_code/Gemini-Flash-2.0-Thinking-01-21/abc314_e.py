import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(n):
        line = list(map(int, sys.stdin.readline().split()))
        c_i = line[0]
        p_i = line[1]
        s_i = line[2:]
        wheels.append({'cost': c_i, 'probabilities': 1.0/p_i, 'scores': s_i, 'p_count': p_i})
    
    dp = [0.0] * m
    
    for current_points in range(m - 1, -1, -1):
        min_expected_cost = float('inf')
        for wheel_index in range(n):
            wheel = wheels[wheel_index]
            cost_wheel = wheel['cost']
            scores = wheel['scores']
            p_count = wheel['p_count']
            expected_future_cost = 0
            for score in scores:
                next_points = current_points + score
                if next_points >= m:
                    future_cost = 0
                else:
                    future_cost = dp[next_points]
                expected_future_cost += future_cost
            expected_cost_wheel = cost_wheel + (expected_future_cost / p_count)
            min_expected_cost = min(min_expected_cost, expected_cost_wheel)
        dp[current_points] = min_expected_cost
        
    print(f"{dp[0]:.12f}")

if __name__ == '__main__':
    solve()