# YOUR CODE HERE
MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def main():
    N = int(input())
    teams = [{i} for i in range(N)]
    wins = [0] * N
    
    for _ in range(N - 1):
        p, q = map(lambda x: int(x) - 1, input().split())
        
        team_p = next(i for i, team in enumerate(teams) if p in team)
        team_q = next(i for i, team in enumerate(teams) if q in team)
        
        a, b = len(teams[team_p]), len(teams[team_q])
        prob_p = a * mod_inverse(a + b) % MOD
        
        for player in teams[team_p]:
            wins[player] = (wins[player] + prob_p) % MOD
        
        teams[team_p] |= teams[team_q]
        teams.pop(team_q)
    
    print(*wins)

if __name__ == "__main__":
    main()