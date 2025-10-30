def solve():
    n = int(input())
    s_list = [input() for _ in range(n)]
    win_counts = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if s_list[i][j] == 'o':
                    win_counts[i] += 1
                    
    player_data = []
    for i in range(n):
        player_data.append({'player_number': i + 1, 'win_count': win_counts[i]})
        
    def compare_players(player1_data, player2_data):
        if player1_data['win_count'] != player2_data['win_count']:
            return player2_data['win_count'] - player1_data['win_count'] # Descending order of win count
        else:
            return player1_data['player_number'] - player2_data['player_number'] # Ascending order of player number for tie
            
    from functools import cmp_to_key
    sorted_players = sorted(player_data, key=cmp_to_key(compare_players))
    
    ranked_player_numbers = [player['player_number'] for player in sorted_players]
    print(*(ranked_player_numbers))

if __name__ == '__main__':
    solve()