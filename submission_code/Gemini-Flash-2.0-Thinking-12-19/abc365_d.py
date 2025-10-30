def solve():
    n = int(input())
    s = input()
    aoki_moves = []
    for char in s:
        if char == 'R':
            aoki_moves.append('R')
        elif char == 'P':
            aoki_moves.append('P')
        else:
            aoki_moves.append('S')
    
    moves_map = {'R': 0, 'P': 1, 'S': 2}
    reverse_moves_map = {0: 'R', 1: 'P', 2: 'S'}
    aoki_moves_index = [moves_map[move] for move in aoki_moves]
    
    def does_takahashi_win(t_move_index, a_move_index):
        if t_move_index == 0: # Rock
            return a_move_index == 2 # Scissors
        elif t_move_index == 1: # Paper
            return a_move_index == 0 # Rock
        else: # Scissors
            return a_move_index == 1 # Paper
            
    def is_not_losing(t_move_index, a_move_index):
        if a_move_index == 0: # Rock
            return t_move_index in [0, 1] # Rock, Paper
        elif a_move_index == 1: # Paper
            return t_move_index in [1, 2] # Paper, Scissors
        else: # Scissors
            return t_move_index in [2, 0] # Scissors, Rock
            
    dp = {}
    
    for first_move_index in range(3):
        if is_not_losing(first_move_index, aoki_moves_index[0]):
            wins = 1 if does_takahashi_win(first_move_index, aoki_moves_index[0]) else 0
            dp[(0, first_move_index)] = wins
        else:
            dp[(0, first_move_index)] = -1 # Mark as invalid starting move
            
    for i in range(1, n):
        for current_move_index in range(3):
            if not is_not_losing(current_move_index, aoki_moves_index[i]):
                dp[(i, current_move_index)] = -1
                continue
            max_prev_wins = -1
            found_prev_move = False
            for prev_move_index in range(3):
                if prev_move_index != current_move_index:
                    if (i - 1, prev_move_index) in dp and dp[(i - 1, prev_move_index)] != -1:
                        max_prev_wins = max(max_prev_wins, dp[(i - 1, prev_move_index)])
                        found_prev_move = True
                        
            if found_prev_move:
                current_game_wins = 1 if does_takahashi_win(current_move_index, aoki_moves_index[i]) else 0
                dp[(i, current_move_index)] = current_game_wins + max_prev_wins
            else:
                dp[(i, current_move_index)] = -1

    max_total_wins = 0
    for last_move_index in range(3):
        if (n - 1, last_move_index) in dp:
            max_total_wins = max(max_total_wins, dp[(n - 1, last_move_index)])
            
    print(max_total_wins)

if __name__ == '__main__':
    solve()