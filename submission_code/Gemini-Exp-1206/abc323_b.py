def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    wins = [0] * n
    for i in range(n):
        for j in range(n):
            if s[i][j] == 'o':
                wins[i] += 1
    
    players = list(range(1, n + 1))
    players.sort(key=lambda x: (-wins[x-1], x))
    
    print(*players)

solve()