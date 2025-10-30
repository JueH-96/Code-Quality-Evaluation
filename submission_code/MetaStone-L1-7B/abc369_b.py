import sys

def main():
    n = int(sys.stdin.readline())
    steps = []
    for _ in range(n):
        a, s = sys.stdin.readline().split()
        a = int(a)
        steps.append((s, a))
    
    INF = float('inf')
    dp_prev = [[INF for _ in range(101)] for _ in range(101)]
    for l in range(101):
        for r in range(101):
            dp_prev[l][r] = 0
    
    for i in range(n):
        s, a = steps[i]
        dp_curr = [[INF for _ in range(101)] for _ in range(101)]
        for l_prev in range(101):
            for r_prev in range(101):
                if dp_prev[l_prev][r_prev] == INF:
                    continue
                if s == 'L':
                    l_curr = a
                    r_curr = r_prev
                    cost = abs(l_curr - l_prev)
                    new_cost = dp_prev[l_prev][r_prev] + cost
                    if new_cost < dp_curr[l_curr][r_curr]:
                        dp_curr[l_curr][r_curr] = new_cost
                else:
                    r_curr = a
                    l_curr = l_prev
                    cost = abs(r_curr - r_prev)
                    new_cost = dp_prev[l_prev][r_prev] + cost
                    if new_cost < dp_curr[l_curr][r_curr]:
                        dp_curr[l_curr][r_curr] = new_cost
        dp_prev = dp_curr
    
    min_cost = INF
    for l in range(101):
        for r in range(101):
            if dp_prev[l][r] < min_cost:
                min_cost = dp_prev[l][r]
    print(min_cost)

if __name__ == '__main__':
    main()