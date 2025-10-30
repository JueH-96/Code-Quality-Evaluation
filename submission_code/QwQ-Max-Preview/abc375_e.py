import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    teams = {1: [], 2: [], 3: []}
    total = 0
    for _ in range(N):
        a = int(input[idx])
        b = int(input[idx+1])
        idx +=2
        teams[a].append(b)
        total += b
    
    # Check if total is divisible by 3
    if total %3 !=0:
        print(-1)
        return
    
    T = total //3
    S = {}
    for i in range(1,4):
        S[i] = sum(teams[i])
    
    D = {1: S[1]-T, 2: S[2]-T, 3: S[3]-T}
    if D[1] + D[2] + D[3] !=0:
        print(-1)
        return
    
    # Precompute flow dicts for each pair of teams
    def compute_flow_dict(x, y):
        team_x = teams[x]
        team_y = teams[y]
        dp = {(0, 0): 0}
        for b in team_x:
            new_dp = {}
            for (s_out, s_in), cnt in dp.items():
                new_s_out = s_out + b
                key = (new_s_out, s_in)
                if key not in new_dp or cnt +1 < new_dp[key]:
                    new_dp[key] = cnt +1
            # Merge new_dp into dp
            for key, val in new_dp.items():
                if key not in dp or val < dp[key]:
                    dp[key] = val
        for b in team_y:
            new_dp = {}
            for (s_out, s_in), cnt in dp.items():
                new_s_in = s_in + b
                key = (s_out, new_s_in)
                if key not in new_dp or cnt +1 < new_dp[key]:
                    new_dp[key] = cnt +1
            # Merge new_dp into dp
            for key, val in new_dp.items():
                if key not in dp or val < dp[key]:
                    dp[key] = val
        flow_dict = {}
        for (s_out, s_in), cnt in dp.items():
            f = s_out - s_in
            if f in flow_dict:
                if cnt < flow_dict[f]:
                    flow_dict[f] = cnt
            else:
                flow_dict[f] = cnt
        return flow_dict
    
    # Compute flow dicts for each pair
    flow_1_2 = compute_flow_dict(1, 2)
    flow_1_3 = compute_flow_dict(1, 3)
    flow_2_3 = compute_flow_dict(2, 3)
    
    sum_team2 = sum(teams[2])
    sum_team3 = sum(teams[3])
    min_f23 = -sum_team3
    max_f23 = sum_team2
    
    min_switches = float('inf')
    for f23 in range(min_f23, max_f23 +1):
        f12 = -D[2] + f23
        f13 = -D[3] - f23
        
        # Check if all flows are present
        if f12 not in flow_1_2:
            continue
        if f13 not in flow_1_3:
            continue
        if f23 not in flow_2_3:
            continue
        
        total = flow_1_2[f12] + flow_1_3[f13] + flow_2_3[f23]
        if total < min_switches:
            min_switches = total
    
    if min_switches == float('inf'):
        print(-1)
    else:
        print(min_switches)

if __name__ == '__main__':
    main()