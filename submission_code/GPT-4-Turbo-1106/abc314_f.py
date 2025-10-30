MOD = 998244353

def modinv(x, mod=MOD):
    return pow(x, mod - 2, mod)

def solve(N, matches):
    win_count = [1] * N
    team_size = [1] * N
    for p, q in matches:
        p -= 1
        q -= 1
        win_prob_p = team_size[p] * modinv(team_size[p] + team_size[q])
        win_prob_q = team_size[q] * modinv(team_size[p] + team_size[q])
        win_count[p] = win_count[p] * win_prob_p % MOD
        win_count[q] = win_count[q] * win_prob_q % MOD
        win_count[p] = (win_count[p] + win_count[q]) % MOD
        team_size[p] += team_size[q]
        team_size[q] = team_size[p]

    for i in range(N):
        print(win_count[i], end=' ')
    print()

def main():
    N = int(input())
    matches = [tuple(map(int, input().split())) for _ in range(N - 1)]
    solve(N, matches)

if __name__ == "__main__":
    main()