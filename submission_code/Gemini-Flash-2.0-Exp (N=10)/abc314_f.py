def modinv(a, m):
    return pow(a, m - 2, m)

def solve():
    n = int(input())
    matches = []
    for _ in range(n - 1):
        matches.append(list(map(int, input().split())))
    
    MOD = 998244353
    
    team_members = [{i} for i in range(1, n + 1)]
    expected_wins = [0] * n
    
    for p, q in matches:
        team_p_idx = -1
        team_q_idx = -1
        for i in range(len(team_members)):
            if p in team_members[i]:
                team_p_idx = i
            if q in team_members[i]:
                team_q_idx = i
        
        team_p = team_members[team_p_idx]
        team_q = team_members[team_q_idx]
        
        len_p = len(team_p)
        len_q = len(team_q)
        
        prob_p_wins = (len_p * modinv(len_p + len_q, MOD)) % MOD
        prob_q_wins = (len_q * modinv(len_p + len_q, MOD)) % MOD
        
        for member in team_p:
            expected_wins[member - 1] = (expected_wins[member - 1] + prob_p_wins) % MOD
        for member in team_q:
            expected_wins[member - 1] = (expected_wins[member - 1] + prob_q_wins) % MOD
        
        team_members[team_p_idx] = team_p.union(team_q)
        team_members.pop(team_q_idx if team_q_idx < team_p_idx else team_q_idx -1)
        
    print(*expected_wins)

solve()