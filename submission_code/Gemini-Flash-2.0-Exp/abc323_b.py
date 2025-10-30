def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    wins = []
    for i in range(n):
        count = 0
        for j in range(n):
            if i != j and s[i][j] == 'o':
                count += 1
        wins.append(count)
    
    players = []
    for i in range(n):
        players.append((wins[i], i + 1))
    
    players.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    
    result = [str(player[1]) for player in players]
    print(" ".join(result))

solve()