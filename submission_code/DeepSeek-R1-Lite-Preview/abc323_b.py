import sys

def main():
    lines = sys.stdin.read().splitlines()
    N = int(lines[0])
    strings = lines[1:N+1]
    players = []
    for i in range(1, N+1):
        s = strings[i-1]
        win_count = s.count('o')
        players.append( (i, win_count) )
    # Sort with key (-win_count, player_number)
    sorted_players = sorted(players, key=lambda x: (-x[1], x[0]))
    # Extract player numbers
    result = [str(player[0]) for player in sorted_players]
    # Print them separated by spaces
    print(' '.join(result))

if __name__ == '__main__':
    main()