import sys

def solve():
    n = int(sys.stdin.readline())
    s_list = [sys.stdin.readline().strip() for _ in range(n)]
    wins_count = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if s_list[i][j] == 'o':
                    wins_count[i] += 1
                    
    player_wins = []
    for i in range(n):
        player_wins.append({'player_number': i + 1, 'wins': wins_count[i]})
        
    def compare_players(player1, player2):
        if player1['wins'] != player2['wins']:
            return player2['wins'] - player1['wins'] # Descending order of wins
        else:
            return player1['player_number'] - player2['player_number'] # Ascending order of player number
            
    from functools import cmp_to_key
    sorted_players = sorted(player_wins, key=cmp_to_key(compare_players))
    
    output_player_numbers = [str(player['player_number']) for player in sorted_players]
    print(*(output_player_numbers))

if __name__ == '__main__':
    solve()