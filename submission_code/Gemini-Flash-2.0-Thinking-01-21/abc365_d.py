def solve():
    n = int(input())
    s = input()
    aoki_moves = list(s)
    takahashi_moves = []
    winning_moves = {'R': 'P', 'P': 'S', 'S': 'R'}
    drawing_moves = {'R': 'R', 'P': 'P', 'S': 'S'}
    
    for i in range(n):
        aoki_move = aoki_moves[i]
        win_move = winning_moves[aoki_move]
        draw_move = drawing_moves[aoki_move]
        
        if i == 0:
            takahashi_moves.append(win_move)
        else:
            last_takahashi_move = takahashi_moves[-1]
            if win_move != last_takahashi_move:
                takahashi_moves.append(win_move)
            else:
                takahashi_moves.append(draw_move)
                
    wins = 0
    for i in range(n):
        t_move = takahashi_moves[i]
        a_move = aoki_moves[i]
        if (t_move == 'R' and a_move == 'S') or \
           (t_move == 'S' and a_move == 'P') or \
           (t_move == 'P' and a_move == 'R'):
            wins += 1
            
    print(wins)

if __name__ == '__main__':
    solve()