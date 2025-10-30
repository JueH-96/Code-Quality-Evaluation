def main():
    import sys
    from collections import defaultdict

    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    incompatibles = [set() for _ in range(N)]
    for _ in range(M):
        a = int(input[ptr]) - 1
        ptr += 1
        b = int(input[ptr]) - 1
        ptr += 1
        incompatibles[a].add(b)
        incompatibles[b].add(a)
    
    # Initialize DP: restricted growth strings
    dp = defaultdict(int)
    dp[tuple([0])] = 1  # Initial state: first player in team 0

    for i in range(1, N):
        new_dp = defaultdict(int)
        for rgs, cnt in dp.items():
            current_teams = max(rgs) + 1 if rgs else 0
            incompatible = incompatibles[i]
            possible_teams = []
            # Consider creating a new team
            possible_teams.append(current_teams)
            # Check existing teams
            for j in range(current_teams):
                can_join = True
                for k in range(len(rgs)):
                    if rgs[k] == j and k in incompatible:
                        can_join = False
                        break
                if can_join:
                    possible_teams.append(j)
            # Update new_dp with all possible_teams
            for t in possible_teams:
                new_rgs = rgs + (t,)
                new_dp[new_rgs] += cnt
        dp = new_dp
    
    answer = 0
    for rgs, cnt in dp.items():
        num_teams = max(rgs) + 1
        if num_teams == T:
            answer += cnt
    print(answer)

if __name__ == "__main__":
    main()