import sys

def main():
    n = int(sys.stdin.readline())
    steps = []
    for _ in range(n):
        a, s = sys.stdin.readline().split()
        steps.append((int(a), s))
    
    INF = 10**9
    # Initialize previous DP table
    dp_prev = [[INF] * 101 for _ in range(101)]
    
    first_A, first_S = steps[0]
    if first_S == 'L':
        a = first_A
        for r in range(1, 101):
            dp_prev[a][r] = 0
    else:
        a = first_A
        for l in range(1, 101):
            dp_prev[l][a] = 0
    
    for i in range(1, n):
        current_A, current_S = steps[i]
        dp_curr = [[INF] * 101 for _ in range(101)]
        for l_prev in range(1, 101):
            for r_prev in range(1, 101):
                if dp_prev[l_prev][r_prev] != INF:
                    if current_S == 'L':
                        new_l = current_A
                        new_r = r_prev
                        cost = abs(new_l - l_prev)
                        total = dp_prev[l_prev][r_prev] + cost
                        if dp_curr[new_l][new_r] > total:
                            dp_curr[new_l][new_r] = total
                    else:
                        new_r = current_A
                        new_l = l_prev
                        cost = abs(new_r - r_prev)
                        total = dp_prev[l_prev][r_prev] + cost
                        if dp_curr[new_l][new_r] > total:
                            dp_curr[new_l][new_r] = total
        dp_prev = dp_curr
    
    min_fatigue = INF
    for l in range(101):
        for r in range(101):
            if dp_prev[l][r] < min_fatigue:
                min_fatigue = dp_prev[l][r]
    
    print(min_fatigue)

if __name__ == "__main__":
    main()