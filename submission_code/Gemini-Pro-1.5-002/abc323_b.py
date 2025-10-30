# YOUR CODE HERE
def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    wins = []
    for i in range(n):
        win_count = 0
        for j in range(n):
            if i != j and s[i][j] == 'o':
                win_count += 1
        wins.append(win_count)
    
    ranked_players = sorted(range(1, n + 1), key=lambda x: (-wins[x - 1], x))
    
    print(*ranked_players)

solve()