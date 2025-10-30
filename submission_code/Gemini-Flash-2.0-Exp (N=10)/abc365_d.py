def solve():
    n = int(input())
    s = input()
    
    wins = 0
    takahashi_moves = []
    
    for i in range(n):
        aoki_move = s[i]
        
        if aoki_move == 'R':
            takahashi_move = 'P'
        elif aoki_move == 'P':
            takahashi_move = 'S'
        else:
            takahashi_move = 'R'
            
        if i > 0 and takahashi_moves[-1] == takahashi_move:
            if aoki_move == 'R':
                takahashi_move = 'S'
            elif aoki_move == 'P':
                takahashi_move = 'R'
            else:
                takahashi_move = 'P'
            
        if aoki_move == 'R' and takahashi_move == 'P':
            wins += 1
        elif aoki_move == 'P' and takahashi_move == 'S':
            wins += 1
        elif aoki_move == 'S' and takahashi_move == 'R':
            wins += 1
        
        takahashi_moves.append(takahashi_move)
    
    print(wins)

solve()