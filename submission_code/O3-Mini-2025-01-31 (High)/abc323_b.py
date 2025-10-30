def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    records = input_data[1:]

    # Calculate wins for each player (player i has record at records[i-1])
    players = []
    for i in range(n):
        wins = records[i].count('o')
        # i+1 because players are numbered starting from 1
        players.append((wins, i+1))

    # Sort players first by descending wins and then by ascending player number
    players.sort(key=lambda x: (-x[0], x[1]))

    # Output the ordered player numbers in a single line separated by space.
    result = ' '.join(str(player) for _, player in players)
    sys.stdout.write(result)

if __name__ == '__main__':
    main()