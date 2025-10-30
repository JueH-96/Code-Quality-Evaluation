import sys
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0]); K = int(data[1]); X = int(data[2])
    T = list(map(int, data[3:3+n]))
    
    prefix = [0]*(n+1)
    for i in range(1, n+1):
        prefix[i] = prefix[i-1] + T[i-1]
    
    NEG_INF = -10**18
    dp = [dict() for _ in range(n+1)]
    dp[0][NEG_INF] = 0
    
    sorted_states = [None] * (n+1)
    prefix_min = [None] * (n+1)
    
    sorted_states[0] = sorted(dp[0].items(), key=lambda x: x[0])
    min_val = 10**30
    min_vals_list = []
    for ship, total_val in sorted_states[0]:
        if total_val < min_val:
            min_val = total_val
        min_vals_list.append(min_val)
    prefix_min[0] = min_vals_list

    for i in range(1, n+1):
        dp_i = {}
        T_i = T[i-1]
        threshold = T_i - X
        
        start_j = max(0, i - K)
        for j in range(start_j, i):
            group_size = i - j
            group_sum = prefix[i] - prefix[j]
            states_list = sorted_states[j]
            if not states_list:
                continue
            pos = bisect.bisect_right(states_list, (threshold, 10**30))
            idx = pos - 1
            if idx >= 0:
                min_total_low = prefix_min[j][idx]
                ship_day = T_i
                dissat = group_size * ship_day - group_sum
                total_new = min_total_low + dissat
                if ship_day not in dp_i or total_new < dp_i[ship_day]:
                    dp_i[ship_day] = total_new
            if pos < len(states_list):
                for k in range(pos, len(states_list)):
                    last_ship_prev, total_prev = states_list[k]
                    ship_day = last_ship_prev + X
                    dissat = group_size * ship_day - group_sum
                    total_new = total_prev + dissat
                    if ship_day not in dp_i or total_new < dp_i[ship_day]:
                        dp_i[ship_day] = total_new
        
        dp[i] = dp_i
        if dp_i:
            sorted_i = sorted(dp_i.items(), key=lambda x: x[0])
            sorted_states[i] = sorted_i
            min_vals_list_i = []
            min_so_far = 10**30
            for ship, total_val in sorted_i:
                if total_val < min_so_far:
                    min_so_far = total_val
                min_vals_list_i.append(min_so_far)
            prefix_min[i] = min_vals_list_i
        else:
            sorted_states[i] = []
            prefix_min[i] = []
    
    if n == 0:
        print(0)
    else:
        print(min(dp[n].values()))

if __name__ == "__main__":
    main()