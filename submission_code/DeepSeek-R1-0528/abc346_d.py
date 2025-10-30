import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s_str = data[1].strip()
    costs = list(map(int, data[2].split()))
    s = [1 if c == '1' else 0 for c in s_str]
    
    INF = 10**18
    dp_prev = [[INF, INF] for _ in range(2)]
    if s[0] == 0:
        dp_prev[0][0] = 0
        dp_prev[1][0] = costs[0]
    else:
        dp_prev[0][0] = costs[0]
        dp_prev[1][0] = 0

    for i in range(1, n):
        dp_curr = [[INF, INF] for _ in range(2)]
        for x_prev in range(2):
            for d_prev in range(2):
                if dp_prev[x_prev][d_prev] == INF:
                    continue
                for x_curr in range(2):
                    cost_here = 0 if s[i] == x_curr else costs[i]
                    if x_prev == x_curr:
                        if d_prev == 0:
                            new_d = 1
                            total_cost = dp_prev[x_prev][d_prev] + cost_here
                            if total_cost < dp_curr[x_curr][new_d]:
                                dp_curr[x_curr][new_d] = total_cost
                    else:
                        new_d = d_prev
                        total_cost = dp_prev[x_prev][d_prev] + cost_here
                        if total_cost < dp_curr[x_curr][new_d]:
                            dp_curr[x_curr][new_d] = total_cost
        dp_prev = dp_curr

    answer = min(dp_prev[0][1], dp_prev[1][1])
    print(answer)

if __name__ == "__main__":
    main()