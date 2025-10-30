def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    courses = []
    index = 1
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        courses.append((x, y))
        index += 2
    
    # Initialize DP: prev_dp[0] is 0 (healthy state), prev_dp[1] is -inf
    prev_dp = [float('-inf'), float('-inf')]
    prev_dp[0] = 0  # initial state is healthy with total 0
    
    for x, y in courses:
        current_dp = [float('-inf'), float('-inf')]
        for s_prev in [0, 1]:
            if prev_dp[s_prev] == float('-inf'):
                continue
            # Option 1: Skip the course
            if current_dp[s_prev] < prev_dp[s_prev]:
                current_dp[s_prev] = prev_dp[s_prev]
            # Option 2: Eat the course
            if x == 0:
                new_state = 0
                new_total = prev_dp[s_prev] + y
                if new_total > current_dp[new_state]:
                    current_dp[new_state] = new_total
            else:
                if s_prev == 0:
                    new_state = 1
                    new_total = prev_dp[s_prev] + y
                    if new_total > current_dp[new_state]:
                        current_dp[new_state] = new_total
        prev_dp = current_dp.copy()
    
    max_total = max(prev_dp[0], prev_dp[1])
    print(max(max_total, 0) if max_total != float('-inf') else 0)

if __name__ == '__main__':
    main()