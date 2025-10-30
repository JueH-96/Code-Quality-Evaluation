import sys

def main():
    N = int(sys.stdin.readline())
    actions = []
    for _ in range(N):
        parts = sys.stdin.readline().split()
        a = int(parts[0])
        s = parts[1]
        actions.append((a, s))
    
    # Initialize DP with all possible starting positions, cost 0
    dp_prev = {}
    for l in range(1, 101):
        for r in range(1, 101):
            dp_prev[(l, r)] = 0
    
    for i in range(N):
        a, s = actions[i]
        dp_current = {}
        for (prev_l, prev_r), cost in dp_prev.items():
            if s == 'L':
                new_l = a
                new_r = prev_r
                added_cost = abs(prev_l - a)
            else:
                new_l = prev_l
                new_r = a
                added_cost = abs(prev_r - a)
            new_cost = cost + added_cost
            key = (new_l, new_r)
            if key in dp_current:
                if new_cost < dp_current[key]:
                    dp_current[key] = new_cost
            else:
                dp_current[key] = new_cost
        dp_prev = dp_current
    
    if dp_prev:
        print(min(dp_prev.values()))
    else:
        print(0)

if __name__ == '__main__':
    main()