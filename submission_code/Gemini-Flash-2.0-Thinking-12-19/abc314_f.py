def solve():
    n = int(input())
    if n == 1:
        print()
        return
    matches = []
    for _ in range(n - 1):
        matches.append(list(map(int, input().split())))
    
    teams = [{i} for i in range(1, n + 1)]
    expected_wins = [0] * n
    mod = 998244353
    
    def power(a, b):
        res = 1
        a %= mod
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % mod
            a = (a * a) % mod
            b //= 2
        return res
        
    def inverse(n):
        return power(n, mod - 2)
        
    for match_index in range(n - 1):
        p_i, q_i = matches[match_index]
        team_p = None
        team_q = None
        for team in teams:
            if p_i in team:
                team_p = team
            if q_i in team:
                team_q = team
        
        size_p = len(team_p)
        size_q = len(team_q)
        prob_p_wins_num = size_p
        prob_q_wins_num = size_q
        denominator = size_p + size_q
        
        prob_p_wins = (prob_p_wins_num * inverse(denominator)) % mod
        prob_q_wins = (prob_q_wins_num * inverse(denominator)) % mod
        
        for player in team_p:
            expected_wins[player-1] = (expected_wins[player-1] + prob_p_wins) % mod
        for player in team_q:
            expected_wins[player-1] = (expected_wins[player-1] + prob_q_wins) % mod
            
        merged_team = team_p.union(team_q)
        next_teams = []
        for team in teams:
            if team != team_p and team != team_q:
                next_teams.append(team)
        next_teams.append(merged_team)
        teams = next_teams
        
    print(*(expected_wins))

if __name__ == '__main__':
    solve()