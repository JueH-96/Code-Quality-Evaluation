def solve():
    n = int(input())
    s = [input() for _ in range(n)]

    wins = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and s[i][j] == 'o':
                wins[i] += 1

    players_data = []
    for i in range(n):
        players_data.append((i + 1, wins[i]))

    players_data.sort(key=lambda item: (-item[1], item[0]))

    ranked_players = [item[0] for item in players_data]
    print(*ranked_players)

solve()