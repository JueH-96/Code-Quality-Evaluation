def solve():
    n = int(input())
    results = [input() for _ in range(n)]
    
    wins = []
    for i in range(n):
        win_count = 0
        for j in range(n):
            if i != j and results[i][j] == 'o':
                win_count += 1
        wins.append((win_count, i + 1))
    
    wins.sort(key=lambda x: (-x[0], x[1]))
    
    print(*[player for _, player in wins])

solve()