# YOUR CODE HERE
def solve():
    n = int(input())
    matches = []
    for _ in range(n - 1):
        matches.append(list(map(int, input().split())))

    mod = 998244353

    def modinv(x):
        return pow(x, mod - 2, mod)

    ans = [0] * n
    for i in range(n):
        wins = 0
        for j in range(1 << (n - 1)):
            teams = [{k} for k in range(1, n + 1)]
            cur_wins = 0
            
            for k in range(n - 1):
                p, q = matches[k]
                team_p = -1
                team_q = -1
                
                for idx, team in enumerate(teams):
                    if p in team:
                        team_p = idx
                    if q in team:
                        team_q = idx
                
                len_p = len(teams[team_p])
                len_q = len(teams[team_q])
                
                win_prob_p = len_p * modinv(len_p + len_q) % mod
                win_prob_q = len_q * modinv(len_p + len_q) % mod

                if (j >> k) & 1:
                    if i + 1 in teams[team_p]:
                        cur_wins += 1
                    teams[team_p] |= teams[team_q]
                    teams.pop(team_q)
                else:
                    if i + 1 in teams[team_q]:
                        cur_wins += 1
                    teams[team_q] |= teams[team_p]
                    teams.pop(team_p)
            wins = (wins + cur_wins) % mod

        ans[i] = wins * modinv(pow(2, n-1, mod)) % mod

    print(*ans)

solve()